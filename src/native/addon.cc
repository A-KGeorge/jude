#include <napi.h>
#include <uv.h>
#include <cstring>
#include <string>
#include <algorithm>
#include <deque>
#include <memory>
#include <mutex>

#include "platform_mmap.h"
#include "segment.h"

/**
 * Optimized string-to-enum conversion.
 * Instead of multiple strcmp calls, we treat the first 8 bytes of the string
 * as a uint64_t (little-endian) to enable a high-speed compiler jump table.
 */
static DType dtype_from_string(const char *s) noexcept
{
    if (!s)
        return DType::UNKNOWN;

    uint64_t magic = 0;
    size_t len = std::strlen(s);
    // Copy up to 8 bytes into our magic number
    std::memcpy(&magic, s, len < 8 ? len : 8);

    switch (magic)
    {
    case 0x323374616f6c66:
        return DType::FLOAT32; // "float32"
    case 0x343674616f6c66:
        return DType::FLOAT64; // "float64"
    case 0x3233746e69:
        return DType::INT32; // "int32"
    case 0x3436746e69:
        return DType::INT64; // "int64"
    case 0x38746e6975:
        return DType::UINT8; // "uint8"
    case 0x38746e69:
        return DType::INT8; // "int8"
    case 0x3631746e6975:
        return DType::UINT16; // "uint16"
    case 0x3631746e69:
        return DType::INT16; // "int16"
    case 0x6c6f6f62:
        return DType::BOOL; // "bool"
    default:
        return DType::UNKNOWN;
    }
}

class SharedTensor : public Napi::ObjectWrap<SharedTensor>
{
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports)
    {
        Napi::Function func = DefineClass(env, "SharedTensor", {
                                                                   InstanceMethod<&SharedTensor::Write>("write"),
                                                                   InstanceMethod<&SharedTensor::Read>("read"),
                                                                   InstanceMethod<&SharedTensor::ReadCopy>("readCopy"),
                                                                   InstanceMethod<&SharedTensor::ReadWait>("readWait"),
                                                                   InstanceMethod<&SharedTensor::ReadCopyWait>("readCopyWait"),
                                                                   InstanceMethod<&SharedTensor::Destroy>("destroy"),
                                                                   InstanceMethod<&SharedTensor::Pin>("pin"),
                                                                   InstanceMethod<&SharedTensor::Unpin>("unpin"),
                                                                   InstanceAccessor<&SharedTensor::ByteCapacity>("byteCapacity"),
                                                                   InstanceAccessor<&SharedTensor::IsPinned>("isPinned"),
                                                               });

        auto *ctor = new Napi::FunctionReference(Napi::Persistent(func));
        env.SetInstanceData<Napi::FunctionReference>(ctor);
        exports.Set("SharedTensor", func);
        return exports;
    }

    SharedTensor(const Napi::CallbackInfo &info) : Napi::ObjectWrap<SharedTensor>(info)
    {
        Napi::Env env = info.Env();
        env_ = env;

        if (info.Length() < 1 || !info[0].IsNumber())
        {
            Napi::TypeError::New(env, "SharedTensor(maxBytes: number)").ThrowAsJavaScriptException();
            return;
        }

        size_t max_bytes = static_cast<size_t>(info[0].As<Napi::Number>().DoubleValue());
        mapped_size_ = DATA_OFFSET + max_bytes;

        // Round up to page boundary
#ifdef _WIN32
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        size_t page_size = static_cast<size_t>(si.dwPageSize);
#else
        size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
        if (page_size == 0)
            page_size = 4096;
#endif
        mapped_size_ = ((mapped_size_ + page_size - 1) / page_size) * page_size;

        mapped_ = platform_mmap(mapped_size_);
        if (mapped_ == nullptr)
        {
            Napi::Error::New(env, "mmap failed").ThrowAsJavaScriptException();
            return;
        }

        // Placement-new the header to initialize atomics/seqlock
        new (mapped_) SegmentHeader();

        uv_loop_t *loop = nullptr;
        if (napi_get_uv_event_loop(env, &loop) != napi_ok || loop == nullptr)
        {
            unmap(false);
            Napi::Error::New(env, "Failed to get uv event loop").ThrowAsJavaScriptException();
            return;
        }

        if (uv_async_init(loop, &read_async_, &SharedTensor::OnReadAsyncWake) != 0)
        {
            unmap(false);
            Napi::Error::New(env, "Failed to initialize uv_async handle").ThrowAsJavaScriptException();
            return;
        }

        read_async_.data = this;
        read_async_initialized_ = true;
    }

    ~SharedTensor() { unmap(false); }

private:
    enum class ReadStatus
    {
        Ok,
        Empty,
        RetryNeeded,
        Destroyed,
    };

    struct PendingRead
    {
        explicit PendingRead(Napi::Env env, bool copy_result)
            : deferred(Napi::Promise::Deferred::New(env)), copy(copy_result)
        {
        }

        Napi::Promise::Deferred deferred;
        bool copy;
    };

    static constexpr uint32_t READ_SPIN_LIMIT = 16;

    Napi::Env env_ = nullptr;
    void *mapped_ = nullptr;
    size_t mapped_size_ = 0;
    bool pinned_ = false; // true after successsful platform_mlock
    uv_async_t read_async_{};
    bool read_async_initialized_ = false;
    bool read_async_closing_ = false;
    std::mutex pending_reads_mu_;
    std::deque<std::shared_ptr<PendingRead>> pending_reads_;

    void close_async_handle()
    {
        if (!read_async_initialized_ || read_async_closing_)
            return;

        // Prevent any in-flight async callback from dereferencing this object.
        read_async_.data = nullptr;
        read_async_closing_ = true;
        uv_close(reinterpret_cast<uv_handle_t *>(&read_async_), nullptr);
    }

    void reject_pending_reads(const char *message)
    {
        Napi::HandleScope scope(env_);

        std::deque<std::shared_ptr<PendingRead>> pending;
        {
            std::lock_guard<std::mutex> lock(pending_reads_mu_);
            pending.swap(pending_reads_);
        }

        for (const auto &item : pending)
        {
            item->deferred.Reject(Napi::Error::New(env_, message).Value());
        }
    }

    void clear_pending_reads()
    {
        std::lock_guard<std::mutex> lock(pending_reads_mu_);
        pending_reads_.clear();
    }

    static void OnReadAsyncWake(uv_async_t *handle)
    {
        auto *self = static_cast<SharedTensor *>(handle->data);
        if (self)
            self->flush_pending_reads();
    }

    void signal_pending_reads()
    {
        if (!read_async_initialized_ || read_async_closing_)
            return;

        bool has_pending = false;
        {
            std::lock_guard<std::mutex> lock(pending_reads_mu_);
            has_pending = !pending_reads_.empty();
        }

        if (has_pending)
            uv_async_send(&read_async_);
    }

    Napi::Value make_result(Napi::Env env, const TensorMeta &meta, uint64_t seq, bool copy)
    {
        Napi::Object result = Napi::Object::New(env);
        Napi::Array shape_arr = Napi::Array::New(env, meta.ndim);
        for (uint32_t i = 0; i < meta.ndim; ++i)
        {
            shape_arr.Set(i, Napi::Number::New(env, static_cast<double>(meta.shape[i])));
        }

        result.Set("shape", shape_arr);
        result.Set("dtype", static_cast<uint32_t>(meta.dtype));
        result.Set("version", Napi::Number::New(env, static_cast<double>(seq)));

        uint8_t *data = segment_data_ptr(mapped_);
        if (copy)
        {
            result.Set("buffer", Napi::Buffer<uint8_t>::Copy(env, data, meta.byte_length));
        }
        else
        {
            result.Set(
                "buffer",
                Napi::ArrayBuffer::New(env, data, meta.byte_length,
                                       [](Napi::Env, void *)
                                       {
                                           // Lifetime is owned by SharedTensor's mmap/unmap.
                                       }));
        }

        return result;
    }

    Napi::Value try_read(Napi::Env env, bool copy, uint32_t spin_limit, ReadStatus &status)
    {
        if (!mapped_)
        {
            status = ReadStatus::Destroyed;
            return env.Null();
        }

        auto *hdr = reinterpret_cast<SegmentHeader *>(mapped_);
        TensorMeta meta;
        uint64_t stable_seq = 0;
        bool got_snapshot = false;

        for (uint32_t attempt = 0; attempt < spin_limit; ++attempt)
        {
            const uint64_t seq0 = hdr->seqlock.sequence.load(std::memory_order_acquire);
            if (seq0 & 1u)
                continue;

            meta = hdr->meta;
            std::atomic_thread_fence(std::memory_order_acquire);

            const uint64_t seq1 = hdr->seqlock.sequence.load(std::memory_order_relaxed);
            if (seq0 != seq1)
                continue;

            stable_seq = seq0;
            got_snapshot = true;
            break;
        }

        if (!got_snapshot)
        {
            status = ReadStatus::RetryNeeded;
            return env.Null();
        }

        if (meta.byte_length == 0)
        {
            status = ReadStatus::Empty;
            return env.Null();
        }

        status = ReadStatus::Ok;
        return make_result(env, meta, stable_seq, copy);
    }

    void flush_pending_reads()
    {
        Napi::HandleScope scope(env_);

        std::deque<std::shared_ptr<PendingRead>> pending;
        {
            std::lock_guard<std::mutex> lock(pending_reads_mu_);
            pending.swap(pending_reads_);
        }

        if (pending.empty())
            return;

        std::deque<std::shared_ptr<PendingRead>> still_waiting;
        for (const auto &item : pending)
        {
            if (!mapped_)
            {
                item->deferred.Reject(Napi::Error::New(env_, "Destroyed").Value());
                continue;
            }

            ReadStatus status = ReadStatus::RetryNeeded;
            Napi::Value value = try_read(env_, item->copy, READ_SPIN_LIMIT, status);
            if (status == ReadStatus::Ok)
            {
                item->deferred.Resolve(value);
            }
            else
            {
                still_waiting.push_back(item);
            }
        }

        if (!still_waiting.empty())
        {
            std::lock_guard<std::mutex> lock(pending_reads_mu_);
            for (auto &item : still_waiting)
                pending_reads_.push_back(std::move(item));
        }
    }

    void unmap(bool reject_waiters)
    {
        if (reject_waiters)
            reject_pending_reads("Destroyed");
        else
            clear_pending_reads();

        close_async_handle();

        if (mapped_)
        {
            reinterpret_cast<SegmentHeader *>(mapped_)->~SegmentHeader();
            platform_munmap(mapped_, mapped_size_);
            mapped_ = nullptr;
        }
    }

    // -----------------------------------------------------------------------
    // pin() → boolean
    //
    // Page-locks the entire mapped region so CUDA DMA can read it directly
    // without a staging copy (cudaMemcpy H2D zero-copy path).
    //
    // Must be called before handing the data pointer to TF_NewTensor.
    // Requires elevated privileges on some systems:
    //   Linux: CAP_IPC_LOCK or sufficient RLIMIT_MEMLOCK
    //   Windows: SeLockMemoryPrivilege or within working-set quota
    // -----------------------------------------------------------------------
    Napi::Value Pin(const Napi::CallbackInfo &info)
    {
        Napi::Env env = info.Env();
        if (!mapped_)
        {
            Napi::Error::New(env, "Destroyed").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        if (pinned_)
            return Napi::Boolean::New(env, true);

        pinned_ = platform_mlock(mapped_, mapped_size_);
        return Napi::Boolean::New(env, pinned_);
    }

    // -----------------------------------------------------------------------
    // unpin() — releases the page lock, allows OS to swap pages again.
    // -----------------------------------------------------------------------
    Napi::Value Unpin(const Napi::CallbackInfo &info)
    {
        if (mapped_ && pinned_)
        {
            platform_munlock(mapped_, mapped_size_);
            pinned_ = false;
        }
        return info.Env().Undefined();
    }

    // -----------------------------------------------------------------------
    // get isPinned → boolean
    // -----------------------------------------------------------------------
    Napi::Value IsPinned(const Napi::CallbackInfo &info)
    {
        return Napi::Boolean::New(info.Env(), pinned_);
    }

    Napi::Value Write(const Napi::CallbackInfo &info)
    {
        Napi::Env env = info.Env();
        if (!mapped_)
        {
            Napi::Error::New(env, "Destroyed").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        // 1. Resolve DType (Fast path)
        DType dtype;
        if (info[1].IsNumber())
        {
            // If the TS wrapper passes an integer ID
            dtype = static_cast<DType>(info[1].As<Napi::Number>().Uint32Value());
        }
        else
        {
            // Fallback for string input
            std::string s = info[1].As<Napi::String>().Utf8Value();
            dtype = dtype_from_string(s.c_str());
        }

        if (dtype == DType::UNKNOWN)
        {
            Napi::TypeError::New(env, "Unsupported dtype").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        // 2. Resolve Buffers/Shape
        Napi::Array shape_arr = info[0].As<Napi::Array>();
        uint32_t ndim = shape_arr.Length();
        if (ndim > MAX_DIMS)
        {
            Napi::RangeError::New(env, "shape rank exceeds MAX_DIMS").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        uint8_t *src_ptr = nullptr;
        size_t src_size = 0;
        if (info[2].IsArrayBuffer())
        {
            auto ab = info[2].As<Napi::ArrayBuffer>();
            src_ptr = reinterpret_cast<uint8_t *>(ab.Data());
            src_size = ab.ByteLength();
        }
        else
        {
            auto ta = info[2].As<Napi::TypedArray>();
            src_ptr = reinterpret_cast<uint8_t *>(ta.ArrayBuffer().Data()) + ta.ByteOffset();
            src_size = ta.ByteLength();
        }

        const size_t capacity = mapped_size_ - DATA_OFFSET;
        if (src_size > capacity)
        {
            Napi::RangeError::New(env, "tensor byte size exceeds segment capacity").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        // 3. Metadata commit with Seqlock
        auto *hdr = reinterpret_cast<SegmentHeader *>(mapped_);
        hdr->seqlock.write_begin(); // Corrected member name

        hdr->meta.ndim = ndim;
        hdr->meta.dtype = dtype;
        hdr->meta.byte_length = src_size;
        for (uint32_t i = 0; i < ndim; ++i)
        {
            hdr->meta.shape[i] = shape_arr.Get(i).As<Napi::Number>().Uint32Value();
        }
        std::memcpy(segment_data_ptr(mapped_), src_ptr, src_size); // Fixed typo

        hdr->seqlock.write_end();
        signal_pending_reads();
        return env.Undefined();
    }

    Napi::Value read_internal(const Napi::CallbackInfo &info, bool copy)
    {
        Napi::Env env = info.Env();
        ReadStatus status = ReadStatus::RetryNeeded;
        return try_read(env, copy, READ_SPIN_LIMIT, status);
    }

    Napi::Value read_wait_internal(const Napi::CallbackInfo &info, bool copy)
    {
        Napi::Env env = info.Env();

        auto pending = std::make_shared<PendingRead>(env, copy);
        ReadStatus status = ReadStatus::RetryNeeded;
        Napi::Value value = try_read(env, copy, READ_SPIN_LIMIT, status);
        if (status == ReadStatus::Ok)
        {
            pending->deferred.Resolve(value);
            return pending->deferred.Promise();
        }
        if (status == ReadStatus::Destroyed)
        {
            pending->deferred.Reject(Napi::Error::New(env, "Destroyed").Value());
            return pending->deferred.Promise();
        }

        {
            std::lock_guard<std::mutex> lock(pending_reads_mu_);
            pending_reads_.push_back(pending);
        }

        return pending->deferred.Promise();
    }

    Napi::Value Read(const Napi::CallbackInfo &info) { return read_internal(info, false); }
    Napi::Value ReadCopy(const Napi::CallbackInfo &info) { return read_internal(info, true); }
    Napi::Value ReadWait(const Napi::CallbackInfo &info) { return read_wait_internal(info, false); }
    Napi::Value ReadCopyWait(const Napi::CallbackInfo &info) { return read_wait_internal(info, true); }

    void Destroy(const Napi::CallbackInfo &info) { unmap(true); }

    Napi::Value ByteCapacity(const Napi::CallbackInfo &info)
    {
        if (!mapped_)
            return Napi::Number::New(info.Env(), 0);
        return Napi::Number::New(info.Env(), static_cast<double>(mapped_size_ - DATA_OFFSET));
    }
};

Napi::Object InitAll(Napi::Env env, Napi::Object exports)
{
    return SharedTensor::Init(env, exports);
}

NODE_API_MODULE(jude_map, InitAll)