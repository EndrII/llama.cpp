from pathlib import Path
import json
r=Path('/home/svt/tuning27b-next');d=Path(json.loads((r/'paths.json').read_text())['source'])/'ggml/src/ggml-cuda'
s=(r/'head_base.cu').read_text()
key='''    const bool nonlinear=strcmp(option,"iq4_nl")==0;'''
code='''    ggml_type external_type = GGML_TYPE_COUNT;
    if (!strcmp(option,"iq3_xxs")) external_type=GGML_TYPE_IQ3_XXS;
    if (!strcmp(option,"iq3_s")) external_type=GGML_TYPE_IQ3_S;
    if (!strcmp(option,"iq4_xs")) external_type=GGML_TYPE_IQ4_XS;
    if (external_type != GGML_TYPE_COUNT) {
        const char * path=getenv("GGML_INT4_HEAD_FILE");
        GGML_ASSERT(path);
        hipStreamCaptureStatus status=hipStreamCaptureStatusNone;
        CUDA_CHECK(hipStreamIsCapturing(stream,&status));GGML_ASSERT(status==hipStreamCaptureStatusNone);
        FILE * f=fopen(path,"rb");GGML_ASSERT(f);
        const size_t row_bytes=ggml_row_size(external_type,tensor->ne[0]);
        const size_t size=row_bytes*tensor->ne[1];
        GGML_ASSERT(fseeko(f,0,SEEK_END)==0);GGML_ASSERT((size_t)ftello(f)==size);rewind(f);
        GGML_ASSERT(size<=ggml_nbytes(tensor));
        std::vector<uint8_t> chunk(16*1024*1024);
        for(size_t pos=0;pos<size;pos+=chunk.size()){
            const size_t n=std::min(chunk.size(),size-pos);GGML_ASSERT(fread(chunk.data(),1,n,f)==n);
            CUDA_CHECK(cudaMemcpyAsync((uint8_t*)tensor->data+pos,chunk.data(),n,cudaMemcpyHostToDevice,stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        fclose(f);
        ggml_tensor * t=const_cast<ggml_tensor*>(tensor);
        t->type=external_type;t->nb[0]=ggml_type_size(external_type);t->nb[1]=row_bytes;
        for(int i=2;i<4;i++)t->nb[i]=t->nb[i-1]*t->ne[i-1];
        GGML_LOG_INFO("int4: experimental output.weight Q6_K -> %s from %s in place; GGUF unchanged\\n",option,path);
        return;
    }
'''
assert s.count(key)==1;s=s.replace(key,code+key)
(d/'int4-wmma.cu').write_text(s);(r/'head_blob.cu').write_text(s)
