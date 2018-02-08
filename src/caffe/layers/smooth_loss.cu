#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/util/im2col.hpp"

#include "caffe/layers/smooth_loss.hpp"
#include "caffe/layers/power_layer.hpp"


namespace caffe {


template <typename Dtype>
void convolve_hv(Blob<Dtype>& dst,Blob<Dtype>& src, Blob<Dtype> & filter, int horiz,int vert ){
  int group_ = src.shape(1);
  int kernel_dim_= filter.count(0);
  
  Blob<Dtype> col_buffer_;
  vector<int> col_buffer_shape_;
  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_*group_);
  col_buffer_shape_.push_back(src.shape(2));
  col_buffer_shape_.push_back(src.shape(3));
  col_buffer_.Reshape(col_buffer_shape_);
 
  int bottom_dim_=src.count(1);
  int top_dim_=dst.count(1);

  int filter_order=(filter.shape(1)-1)/2;
  int col_offset_ = kernel_dim_*dst.shape(2)*dst.shape(3);
  int out_offset_ = dst.shape(2)*dst.shape(3);
  
if(horiz&&vert){
  for(int n=0; n< src.shape(0);++n){

  Blob<Dtype> temp;
  vector<int> temp_shape(3);
  temp_shape[0]=src.shape(1);
  temp_shape[1]=src.shape(2);
  temp_shape[2]=src.shape(3);
  temp.Reshape(temp_shape);
  int temp_offset_=dst.shape(2)*dst.shape(3);
  Dtype* temp_data= temp.mutable_gpu_data();
  im2col_gpu(src.gpu_data()+n*bottom_dim_,src.shape(1),src.shape(2),src.shape(3),filter.shape(0),filter.shape(1),0,filter_order,1,1,1,1,col_buffer_.mutable_gpu_data());
  for(int g=0;g<group_;++g){
     caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,1,dst.shape(2)*dst.shape(3),kernel_dim_,(Dtype)1.,filter.gpu_data(),col_buffer_.gpu_data()+col_offset_*g,(Dtype)0.,temp_data+g*temp_offset_);
     }
  im2col_gpu(temp_data,src.shape(1),src.shape(2),src.shape(3),filter.shape(1),filter.shape(0),filter_order,0,1,1,1,1,col_buffer_.mutable_gpu_data());
  for(int g=0;g<group_;++g){
     caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,1,dst.shape(2)*dst.shape(3),kernel_dim_,(Dtype)1.,filter.gpu_data(),col_buffer_.gpu_data()+col_offset_*g,(Dtype)0.,dst.mutable_gpu_data()+n*top_dim_+g*out_offset_);
     }
   }
}else if(horiz && !vert){
 for(int n=0; n< src.shape(0);++n){
  
  im2col_gpu(src.gpu_data()+n*bottom_dim_,src.shape(1),src.shape(2),src.shape(3),filter.shape(0),filter.shape(1),0,filter_order,1,1,1,1,col_buffer_.mutable_gpu_data());

  for(int g=0;g<group_;++g){
     caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,1,dst.shape(2)*dst.shape(3),kernel_dim_,(Dtype)1.,filter.gpu_data(),col_buffer_.gpu_data()+col_offset_*g,(Dtype)0.,dst.mutable_gpu_data()+n*top_dim_+g*out_offset_);
   }
  }   
}else if(!horiz&&vert){
  for(int n=0;n< src.shape(0);++n){
    
  im2col_gpu(src.gpu_data()+n*bottom_dim_,src.shape(1),src.shape(2),src.shape(3),filter.shape(1),filter.shape(0),filter_order,0,1,1,1,1,col_buffer_.mutable_gpu_data());
     
  for(int g=0;g<group_;++g){
     caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,1,dst.shape(2)*dst.shape(3),kernel_dim_,(Dtype)1.,filter.gpu_data(),col_buffer_.gpu_data()+col_offset_*g,(Dtype)0.,dst.mutable_gpu_data()+n*top_dim_+g*out_offset_);

   }
  }
 }
}






template <typename Dtype>
__global__ void ComputeSign(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? Dtype(1) : Dtype(-1);
  }
} 

template <typename Dtype>
__global__ void FindNotNaNs(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index]==in[index] ? Dtype(1) : Dtype(0);
  }
} 

template <typename Dtype>
__global__ void KillNaNs(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index]==in[index] ? in[index] : Dtype(0);
  }
}

template <typename Dtype>
__global__ void KillMasked(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > Dtype(0.5) ? out[index] : Dtype(0);
//     out[index] = out[index]==out[index] ? out[index] : Dtype(0);
//     out[index] = out[index]>1e3 ? 0 : out[index];
//     out[index] = out[index]<-1e3 ? 0 : out[index];
  }
}

template <typename Dtype>
__global__ void multiplyweight(const int n,const Dtype* match_weight_, Dtype* sqrt_output_){
   CUDA_KERNEL_LOOP(index, n) {

    sqrt_output_[index]=sqrt_output_[index]*match_weight_[index];

  }
}



template <typename Dtype>
__global__ void KillMaskedAcrossChannels(const int n, const int width_height, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int mask_idx = index % width_height;
    out[index] = in[mask_idx] > Dtype(0.5) ? out[index] : Dtype(0);
  }
}

template <typename Dtype>
__global__ void MaskPlateauValues(const int n, const Dtype* in, Dtype* out, Dtype plateau) {
  CUDA_KERNEL_LOOP(index, n) {
    if(fabs(in[index]) < plateau) out[index] = Dtype(0); // Mask out plateau values and keep other as is
  }
} 

template <typename Dtype>
__global__ void MaskPlateauValuesInitial(const int n, const Dtype* in, Dtype* out, Dtype plateau) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = (fabs(in[index]) < plateau) ? Dtype(0) : Dtype(1);
  }
} 




template <typename Dtype>
__global__ void compute_lum(const int nthreads,int H,int W,const Dtype* image ,Dtype* lum){
  CUDA_KERNEL_LOOP(index, nthreads) {
       int t=index % W;
       int s= (index/W) % H;
       int i= index/(W*H);
    
     lum[index]=(0.299f*image[i*(3*W*H)+s*W+t]+ 0.587f*image[i*(3*W*H)+W*H+s*W+t]+ 0.114f*image[i*(3*W*H)+2*W*H+s*W+t])/255.0f;

  }
}




template <typename Dtype>
__global__ void compute_smoothweight(const int nthreads,int H,int W, const Dtype* lum_x, const Dtype* lum_y, Dtype* smoothweight_){

  CUDA_KERNEL_LOOP(index, nthreads) {   
   smoothweight_[index]=-5.0f*sqrt(lum_x[index]*lum_x[index]+lum_y[index]*lum_y[index]);
   smoothweight_[index]=0.5f*expf(smoothweight_[index]);
  }
}

template <typename Dtype>
__global__ void compute_finalweight(const int nthreads,int H,int W,const Dtype* smoothweight_,Dtype* smoothweight_h_, Dtype* smoothweight_v_){
  CUDA_KERNEL_LOOP(index, nthreads) {
       int t=index % W;
       int s= (index/W) % H;
      // int i= index/(W*H);
  if(t==W-1){
   smoothweight_h_[index]=smoothweight_[index];
  }else{
   smoothweight_h_[index]=smoothweight_[index]+smoothweight_[index+1];
   }
  if(s==H-1){
    smoothweight_v_[index]=smoothweight_[index];
    } else{
    smoothweight_v_[index]=smoothweight_[index]+smoothweight_[index+W];
    }
 }
}

template <typename Dtype>
__global__ void compute_flow_gradient(const int nthreads2, int H, int W,int C,const Dtype* flow, Dtype* gx1 ,Dtype* gy1 ,Dtype* gx2 ,Dtype* gy2){
  CUDA_KERNEL_LOOP(index, nthreads2) {
       int t=index % W;
       int s= (index/W) % H;
       int j= (index/W*H) % C;
       int i= index/(W*H*C);

  if(j==0){
   
           if(t==W-1){
                gx1[i*(W*H*C)+s*W+t]=0;
                gx2[i*(W*H*C)+s*W+t]=0;
             }else{

                gx1[i*(W*H*C)+s*W+t]=flow[i*(W*H*C)+s*W+t+1]-flow[i*(W*H*C)+s*W+t];
                
                    if(t==0){
                         gx2[i*(W*H*C)+s*W+t]=0;
                            }
                            else{
                         gx2[i*(W*H*C)+s*W+t]=-0.5f*flow[i*(W*H*C)+s*W+t-1]+0.5f*flow[i*(W*H*C)+s*W+t+1];
                                 }
             }


          if(s==H-1){

              gy1[i*(W*H*C)+s*W+t]=0;
              gy2[i*(W*H*C)+s*W+t]=0;

           }else{
               if(s==0){
                 gy2[i*(W*H*C)+s*W+t]=0;
                }else{
                 gy2[i*(W*H*C)+s*W+t]=0.5f*flow[i*(W*H*C)+s*W+t+W]-0.5f*flow[i*(W*H*C)+s*W+t-W];
                }
             
              gy1[i*(W*H*C)+s*W+t]=flow[i*(W*H*C)+s*W+t+W]-flow[i*(W*H*C)+s*W+t];
               }



    }else{



           if(t==W-1){
                gx1[i*(W*H*C)+W*H+s*W+t]=0;
                gx2[i*(W*H*C)+W*H+s*W+t]=0;
             }else{

                gx1[i*(W*H*C)+W*H+s*W+t]=flow[i*(W*H*C)+W*H+s*W+t+1]-flow[i*(W*H*C)+W*H+s*W+t];
                
                    if(t==0){
                         gx2[i*(W*H*C)+W*H+s*W+t]=0;
                            }
                            else{
                         gx2[i*(W*H*C)+W*H+s*W+t]=-0.5f*flow[i*(W*H*C)+W*H+s*W+t-1]+0.5f*flow[i*(W*H*C)+W*H+s*W+t+1];
                                 }
             }


          if(s==H-1){

              gy1[i*(W*H*C)+W*H+s*W+t]=0;
              gy2[i*(W*H*C)+W*H+s*W+t]=0;

           }else{
               if(s==0){
                 gy2[i*(W*H*C)+W*H+s*W+t]=0;
                }else{
                 gy2[i*(W*H*C)+W*H+s*W+t]=0.5f*flow[i*(W*H*C)+W*H+s*W+t+W]-0.5f*flow[i*(W*H*C)+W*H+s*W+t-W];
                }
             
              gy1[i*(W*H*C)+W*H+s*W+t]=flow[i*(W*H*C)+W*H+s*W+t+W]-flow[i*(W*H*C)+W*H+s*W+t];
               }

    }


 }
}

template <typename Dtype>
__global__ void compute_loss_square(const int nthreads,int H,int W,const Dtype* gx1,const Dtype* gy1 , const Dtype* gx2 ,const Dtype* gy2 , Dtype* horiz_loss_, Dtype * verti_loss_){
  CUDA_KERNEL_LOOP(index, nthreads) {
       int t=index % W;
       int s= (index/W) % H;
       
       int i= index/(W*H);

            if(t==W-1){
    horiz_loss_[index]=gx1[i*(W*H*2)+s*W+t]*gx1[i*(W*H*2)+s*W+t]+gy2[i*(2*H*W)+s*W+t]*gy2[i*(2*H*W)+s*W+t]+gx1[i*(H*W*2)+H*W+s*W+t]*gx1[i*(H*W*2)+H*W+s*W+t]+gy2[i*(W*H*2)+H*W+s*W+t]*gy2[i*(W*H*2)+H*W+s*W+t];
             }else{
    horiz_loss_[index]=gx1[i*(W*H*2)+s*W+t]*gx1[i*(W*H*2)+s*W+t]+0.25f*(gy2[i*(2*H*W)+s*W+t]+gy2[i*(2*H*W)+s*W+t+1])*(gy2[i*(2*H*W)+s*W+t]+gy2[i*(2*H*W)+s*W+t+1])+gx1[i*(H*W*2)+H*W+s*W+t]*gx1[i*(H*W*2)+H*W+s*W+t]+0.25f*(gy2[i*(W*H*2)+H*W+s*W+t]+gy2[i*(W*H*2)+H*W+s*W+t+1])*(gy2[i*(W*H*2)+H*W+s*W+t]+gy2[i*(W*H*2)+H*W+s*W+t+1]);
             }

           if(s==H-1){
    verti_loss_[index]=gy1[i*(H*W*2)+s*W+t]*gy1[i*(H*W*2)+s*W+t]+gy1[i*(W*H*2)+W*H+s*W+t]*gy1[i*(W*H*2)+W*H+s*W+t]+gx2[i*(W*H*2)+s*W+t]*gx2[i*(W*H*2)+s*W+t]+gx2[i*(W*H*2)+W*H+s*W+t]*gx2[i*(W*H*2)+W*H+s*W+t];
             }else{

    verti_loss_[index]=gy1[i*(H*W*2)+s*W+t]*gy1[i*(H*W*2)+s*W+t]+gy1[i*(W*H*2)+W*H+s*W+t]*gy1[i*(W*H*2)+W*H+s*W+t]+ 0.25f*(gx2[i*(W*H*2)+s*W+t]+gx2[i*(W*H*2)+s*W+t+W])*(gx2[i*(W*H*2)+s*W+t]+gx2[i*(W*H*2)+s*W+t+W])+0.25f*(gx2[i*(W*H*2)+W*H+s*W+t]+gx2[i*(W*H*2)+W*H+s*W+t+W])*(gx2[i*(W*H*2)+W*H+s*W+t]+gx2[i*(W*H*2)+W*H+s*W+t+W]);

              }



  }
}


template <typename Dtype>
__global__ void computeflowgradient_diff(const int n,int H,int W, const Dtype* horiz_loss_,const Dtype* verti_loss_ ,const Dtype* gx1_data_,const Dtype* gy1_data_ , const Dtype* gx2_data_ , const Dtype* gy2_data_, Dtype* gx1_diff_, Dtype* gy1_diff_, Dtype* d2_tmp_diff_){
  CUDA_KERNEL_LOOP(index, n) {
       int t=index % W;
       int s= (index/W) % H;
       
       int i= index/(W*H);

  gx1_diff_[i*(W*H*2)+s*W+t]=2.0f*gx1_data_[i*(W*H*2)+s*W+t]*horiz_loss_[index];
  gx1_diff_[i*(W*H*2)+s*W+t+W*H]=2.0f*gx1_data_[i*(W*H*2)+s*W+t+W*H]*horiz_loss_[index];
  
  gy1_diff_[i*(W*H*2)+s*W+t]=2.0f*gy1_data_[i*(W*H*2)+s*W+t]*verti_loss_[index];
  gy1_diff_[i*(W*H*2)+s*W+t+H*W]=2.0f*gy1_data_[i*(W*H*2)+s*W+t+H*W]*verti_loss_[index];

  if(t==W-1){
     d2_tmp_diff_[i*(W*H*4*2)+2*W*H*2+s*W*2+t*2]=2.0f*gy2_data_[i*(2*W*H)+s*W+t]* horiz_loss_[index];
     d2_tmp_diff_[i*(W*H*4*2)+3*W*H*2+s*W*2+t*2]=2.0f*gy2_data_[i*(2*W*H)+s*W+t+W*H]* horiz_loss_[index];
     
   }else{

     d2_tmp_diff_[i*(W*H*4*2)+2*W*H*2+s*W*2+t*2]=gy2_data_[i*(2*W*H)+s*W+t]*horiz_loss_[index];
     d2_tmp_diff_[i*(W*H*4*2)+2*W*H*2+s*W*2+(t+1)*2+1]=gy2_data_[i*(2*W*H)+s*W+t+1]*horiz_loss_[index];

     d2_tmp_diff_[i*(W*H*4*2)+3*W*H*2+s*W*2+t*2]=gy2_data_[i*(2*W*H)+s*W+t+H*W]*horiz_loss_[index];
     d2_tmp_diff_[i*(W*H*4*2)+3*W*H*2+s*W*2+(t+1)*2+1]=gy2_data_[i*(2*W*H)+s*W+t+1+H*W]*horiz_loss_[index];
   }
  
  if(s==H-1){

     d2_tmp_diff_[i*(4*W*H*2)+s*W*2+t*2]=2.0f*gx2_data_[i*(W*H*2)+s*W+t]*verti_loss_[index];


     d2_tmp_diff_[i*(4*W*H*2)+W*H*2+s*W*2+t*2]=2.0f*gx2_data_[i*(W*H*2)+W*H+s*W+t]*verti_loss_[index];

   }else{

    d2_tmp_diff_[i*(4*W*H*2)+s*W*2+t*2]=gx2_data_[i*(W*H*2)+s*W+t]*verti_loss_[index];
    d2_tmp_diff_[i*(4*W*H*2)+(s+1)*W*2+t*2+1]=gx2_data_[i*(W*H*2)+(s+1)*W+t]*verti_loss_[index];

    d2_tmp_diff_[i*(4*W*H*2)+W*H*2+s*W*2+t*2]=gx2_data_[i*(W*H*2)+W*H+s*W+t]*verti_loss_[index];
    d2_tmp_diff_[i*(4*W*H*2)+W*H*2+(s+1)*W*2+t*2+1]=gx2_data_[i*(W*H*2)+W*H+(s+1)*W+t]*verti_loss_[index];
   }

 }
}



template <typename Dtype>
__global__ void computeflowgradient2_diff(const int n, int H,int W, const Dtype* d2_tmp_ ,Dtype* gx2_ , Dtype* gy2_ ){

CUDA_KERNEL_LOOP(index, n) {
       int t=index % W;
       int s= (index/W) % H;
       
       int i= index/(W*H);

 gx2_[i*(2*W*H)+W*s+t]=d2_tmp_[i*(4*W*H*2)+s*W*2+t*2]+d2_tmp_[i*(4*W*H*2)+s*W*2+t*2+1];
 gx2_[i*(2*W*H)+H*W+W*s+t]=d2_tmp_[i*(4*W*H*2)+W*H*2+s*W*2+t*2]+d2_tmp_[i*(4*W*H*2)+W*H*2+s*W*2+t*2+1];

 gy2_[i*(2*W*H)+s*W+t]=d2_tmp_[i*(4*W*H*2)+2*W*H*2+s*W*2+t*2]+d2_tmp_[i*(4*W*H*2)+2*W*H*2+s*W*2+t*2+1];
 gy2_[i*(2*W*H)+W*H+s*W+t]=d2_tmp_[i*(4*W*H*2)+3*W*H*2+s*W*2+t*2]+d2_tmp_[i*(4*W*H*2)+3*W*H*2+s*W*2+t*2+1];
 }
}

template <typename Dtype>
__global__ void computeflow_diff1(const int n,int W,int H,const Dtype* gx1_,const Dtype* gy1_,Dtype* d2_tmp_){

CUDA_KERNEL_LOOP(index, n) {
       int t=index % W;
       int s= (index/W) % H;
       
       int i= index/(W*H);

   if(t != W-1){

      d2_tmp_[i*(4*H*W*2)+s*W*2+t*2]=-gx1_[i*(2*W*H)+s*W+t];
      d2_tmp_[i*(4*H*W*2)+s*W*2+(t+1)*2+1]=gx1_[i*(2*W*H)+s*W+t];
      
      d2_tmp_[i*(4*H*W*2)+H*W*2+s*W*2+t*2]=-gx1_[i*(2*W*H)+W*H+s*W+t];
      d2_tmp_[i*(4*W*H*2)+H*W*2+s*W*2+(t+1)*2+1]=gx1_[i*(2*W*H)+W*H+s*W+t];

            }
    if(s!=H-1){

      d2_tmp_[i*(4*H*W*2)+2*H*W*2+s*W*2+t*2]=-gy1_[i*(W*H*2)+s*W+t];
      d2_tmp_[i*(4*H*W*2)+2*H*W*2+(s+1)*W*2+t*2+1]=gy1_[i*(W*H*2)+s*W+t];

      d2_tmp_[i*(4*H*W*2)+3*H*W*2+s*W*2+t*2]=-gy1_[i*(W*H*2)+W*H+s*W+t];
      d2_tmp_[i*(4*H*W*2)+3*H*W*2+(s+1)*W*2+t*2+1]=gy1_[i*(W*H*2)+W*H+s*W+t];

            }

 }
}


template <typename Dtype>
__global__ void computeflow_diff2(const int n, int W, int H,const Dtype* gx2_,const Dtype* gy2_,Dtype* d2_tmp_){

CUDA_KERNEL_LOOP(index, n) {
       int t=index % W;
       int s= (index/W) % H;
       
       int i= index/(W*H);

   if( (t!=W-1) && (t!=0)){

      d2_tmp_[i*(4*H*W*2)+s*W*2+(t-1)*2]=-0.5f*gx2_[i*(2*W*H)+s*W+t];
      d2_tmp_[i*(4*H*W*2)+s*W*2+(t+1)*2+1]=0.5f*gx2_[i*(2*W*H)+s*W+t];
      
      d2_tmp_[i*(4*H*W*2)+H*W*2+s*W*2+(t-1)*2]=-0.5f*gx2_[i*(2*W*H)+W*H+s*W+t];
      d2_tmp_[i*(4*W*H*2)+H*W*2+s*W*2+(t+1)*2+1]=0.5f*gx2_[i*(2*W*H)+W*H+s*W+t];

            }
    if((s != H-1)&&(s!= 0)){

      d2_tmp_[i*(4*H*W*2)+2*H*W*2+(s-1)*W*2+t*2]=-0.5f*gy2_[i*(W*H*2)+s*W+t];
      d2_tmp_[i*(4*H*W*2)+2*H*W*2+(s+1)*W*2+t*2+1]=0.5f*gy2_[i*(W*H*2)+s*W+t];

      d2_tmp_[i*(4*H*W*2)+3*H*W*2+(s-1)*W*2+t*2]=-0.5f*gy2_[i*(W*H*2)+W*H+s*W+t];
      d2_tmp_[i*(4*H*W*2)+3*H*W*2+(s+1)*W*2+t*2+1]=0.5f*gy2_[i*(W*H*2)+W*H+s*W+t];

            }

 }
}


template <typename Dtype>
__global__ void sumflow_diff(const int n ,int W,int H, const Dtype* d2_tmp_ , Dtype* flow){
CUDA_KERNEL_LOOP(index, n) {
       int t=index % W;
       int s= (index/W) % H;
       
       int i= index/(W*H);

       flow[i*(2*W*H)+s*W+t]=flow[i*(2*W*H)+s*W+t]+ d2_tmp_[i*(4*W*H*2)+s*W*2+t*2]+d2_tmp_[i*(4*W*H*2)+s*W*2+t*2+1]+d2_tmp_[i*(4*W*H*2)+2*W*H*2+s*W*2+t*2]+d2_tmp_[i*(4*W*H*2)+2*W*H*2+s*W*2+t*2+1];
       
       flow[i*(2*W*H)+W*H+s*W+t]=flow[i*(2*W*H)+W*H+s*W+t]+ d2_tmp_[i*(4*W*H*2)+W*H*2+s*W*2+t*2]+d2_tmp_[i*(4*W*H*2)+W*H*2+s*W*2+t*2+1]+d2_tmp_[i*(4*W*H*2)+3*W*H*2+s*W*2+t*2]+d2_tmp_[i*(4*W*H*2)+3*W*H*2+s*W*2+t*2+1];


 }
}





template <typename Dtype>
void SmoothLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{

  const int nthreads=N*1*H*W;
 
  

  compute_lum<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
			CAFFE_CUDA_NUM_THREADS>>>(nthreads,H,W,bottom[1]->gpu_data(),lum.mutable_gpu_data());

  

  convolve_hv<Dtype>(lum_x_,lum,image_filter,1,0); 
  convolve_hv<Dtype>(lum_y_,lum,image_filter,0,1); 


  compute_smoothweight<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
			CAFFE_CUDA_NUM_THREADS>>>(nthreads,H,W,lum_x_.gpu_data(),lum_y_.gpu_data(),smoothweight_.mutable_gpu_data());


  compute_finalweight<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
			CAFFE_CUDA_NUM_THREADS>>>(nthreads,H,W,smoothweight_.gpu_data(),smoothweight_h_.mutable_gpu_data(),smoothweight_v_.mutable_gpu_data());
 

  Dtype dot1,dot2, loss;
  
  const int nthreads2=N*C*H*W;

  compute_flow_gradient<Dtype><<<CAFFE_GET_BLOCKS(nthreads2),
			CAFFE_CUDA_NUM_THREADS>>>(nthreads2,H,W,C,bottom[0]->gpu_data(),gx1_.mutable_gpu_data(),gy1_.mutable_gpu_data(),gx2_.mutable_gpu_data(),gy2_.mutable_gpu_data());  
  
 
  compute_loss_square<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
			CAFFE_CUDA_NUM_THREADS>>>(nthreads,H,W,gx1_.gpu_data(),gy1_.gpu_data(),gx2_.gpu_data(),gy2_.gpu_data(),horiz_loss_.mutable_gpu_data(),verti_loss_.mutable_gpu_data());
   
  

       
    // Mask plateau in summed blob (only one channel):
    if(this->layer_param_.smooth_loss_param().plateau() > 0) {
      float plateau_val_squared = this->layer_param_.smooth_loss_param().plateau() * this->layer_param_.smooth_loss_param().plateau();
      MaskPlateauValuesInitial<Dtype><<<CAFFE_GET_BLOCKS(horiz_loss_.count()), CAFFE_CUDA_NUM_THREADS>>>(
          horiz_loss_.count(), horiz_loss_.gpu_data(), plateau1_l2_.mutable_gpu_data(), plateau_val_squared);
      cudaDeviceSynchronize();
      CUDA_POST_KERNEL_CHECK;
      
      KillMasked<Dtype><<<CAFFE_GET_BLOCKS(horiz_loss_.count()), CAFFE_CUDA_NUM_THREADS>>>(
          horiz_loss_.count(), plateau1_l2_.gpu_data(), horiz_loss_.mutable_gpu_data());
      cudaDeviceSynchronize();
      CUDA_POST_KERNEL_CHECK;


      MaskPlateauValuesInitial<Dtype><<<CAFFE_GET_BLOCKS(verti_loss_.count()), CAFFE_CUDA_NUM_THREADS>>>(
          verti_loss_.count(), verti_loss_.gpu_data(), plateau2_l2_.mutable_gpu_data(), plateau_val_squared);
      cudaDeviceSynchronize();
      CUDA_POST_KERNEL_CHECK;
      
      KillMasked<Dtype><<<CAFFE_GET_BLOCKS(verti_loss_.count()), CAFFE_CUDA_NUM_THREADS>>>(
          verti_loss_.count(), plateau2_l2_.gpu_data(), verti_loss_.mutable_gpu_data());
      cudaDeviceSynchronize();
      CUDA_POST_KERNEL_CHECK;

    }
    
  
      sqrt1_layer_->Forward(sqrt1_bottom_vec_, sqrt1_top_vec_);
      sqrt2_layer_->Forward(sqrt2_bottom_vec_, sqrt2_top_vec_);

 
      multiplyweight<Dtype><<<CAFFE_GET_BLOCKS(sqrt1_output_.count()), CAFFE_CUDA_NUM_THREADS>>>(
            sqrt1_output_.count(),smoothweight_h_.gpu_data(),sqrt1_output_.mutable_gpu_data());
      cudaDeviceSynchronize();
      CUDA_POST_KERNEL_CHECK;

      multiplyweight<Dtype><<<CAFFE_GET_BLOCKS(sqrt2_output_.count()), CAFFE_CUDA_NUM_THREADS>>>(
            sqrt2_output_.count(),smoothweight_v_.gpu_data(),sqrt2_output_.mutable_gpu_data());
      cudaDeviceSynchronize();
      CUDA_POST_KERNEL_CHECK;


      if (this->layer_param_.smooth_loss_param().normalize_by_num_entries()) {    
           
            normalize_coeff_=N*H*W;
      } else {
           normalize_coeff_ = N;
     }

    // Note sign_ is set to all ones in Reshape

    caffe_gpu_dot(sqrt1_output_.count(), sqrt1_output_.gpu_data(), sign_.gpu_data(), &dot1);
    caffe_gpu_dot(sqrt2_output_.count(), sqrt2_output_.gpu_data(), sign_.gpu_data(), &dot2);
  
    loss = (dot1+dot2) / normalize_coeff_; 
    top[0]->mutable_cpu_data()[0] = loss;
 
}

template <typename Dtype>
void SmoothLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{  
  bool prop_down = propagate_down[0];
  if(bottom.size() > 1) {prop_down |= propagate_down[1];
                         prop_down |= propagate_down[2];
                         }
  
  
  
  if (prop_down) {
    const Dtype alpha = top[0]->cpu_diff()[0] ;
    
      vector<bool> prop_down(1,true);
      caffe_gpu_axpby(sqrt1_output_.count(), alpha, sign_.gpu_data(),                   
          Dtype(0), sqrt1_output_.mutable_gpu_diff());
       
      multiplyweight<Dtype><<<CAFFE_GET_BLOCKS(sqrt1_output_.count()), CAFFE_CUDA_NUM_THREADS>>>(
            sqrt1_output_.count(),smoothweight_h_.gpu_data(),sqrt1_output_.mutable_gpu_diff());
      cudaDeviceSynchronize();
      CUDA_POST_KERNEL_CHECK;
 
      caffe_gpu_axpby(sqrt2_output_.count(), alpha, sign_.gpu_data(),                   
          Dtype(0), sqrt2_output_.mutable_gpu_diff());
     
      multiplyweight<Dtype><<<CAFFE_GET_BLOCKS(sqrt1_output_.count()), CAFFE_CUDA_NUM_THREADS>>>(
            sqrt2_output_.count(),smoothweight_v_.gpu_data(),sqrt2_output_.mutable_gpu_diff());
      cudaDeviceSynchronize();
      CUDA_POST_KERNEL_CHECK;

 



      sqrt1_layer_->Backward(sqrt1_top_vec_, prop_down, sqrt1_bottom_vec_);
      sqrt2_layer_->Backward(sqrt2_top_vec_, prop_down, sqrt2_bottom_vec_);

 
      
      if(this->layer_param_.smooth_loss_param().plateau() > 0) {
        KillMasked<Dtype><<<CAFFE_GET_BLOCKS(horiz_loss_.count()), CAFFE_CUDA_NUM_THREADS>>>(
              horiz_loss_.count(), plateau1_l2_.gpu_data(), horiz_loss_.mutable_gpu_diff());
        cudaDeviceSynchronize();
        CUDA_POST_KERNEL_CHECK;


        KillMasked<Dtype><<<CAFFE_GET_BLOCKS(verti_loss_.count()), CAFFE_CUDA_NUM_THREADS>>>(
              verti_loss_.count(), plateau2_l2_.gpu_data(), verti_loss_.mutable_gpu_diff());
        cudaDeviceSynchronize();
        CUDA_POST_KERNEL_CHECK;
       }
   
     caffe_gpu_set(d2_tmp_.count() , Dtype(0), d2_tmp_.mutable_gpu_diff());
      
     computeflowgradient_diff<Dtype><<<CAFFE_GET_BLOCKS(verti_loss_.count()), CAFFE_CUDA_NUM_THREADS>>>(verti_loss_.count(),H,W,horiz_loss_.gpu_diff(),verti_loss_.gpu_diff(),gx1_.gpu_data(),gy1_.gpu_data(),gx2_.gpu_data(),gy2_.gpu_data(),gx1_.mutable_gpu_diff(),gy1_.mutable_gpu_diff(),d2_tmp_.mutable_gpu_diff());
     
     computeflowgradient2_diff<Dtype><<<CAFFE_GET_BLOCKS(verti_loss_.count()), CAFFE_CUDA_NUM_THREADS>>>(verti_loss_.count(),H,W,d2_tmp_.gpu_diff(),gx2_.mutable_gpu_diff(),gy2_.mutable_gpu_diff());
   
    caffe_gpu_set(d2_tmp_.count() , Dtype(0), d2_tmp_.mutable_gpu_diff());
     caffe_gpu_set(bottom[0]->count(),Dtype(0),bottom[0]->mutable_gpu_diff());
 
     computeflow_diff1<Dtype><<<CAFFE_GET_BLOCKS(verti_loss_.count()), CAFFE_CUDA_NUM_THREADS>>>(verti_loss_.count(),W,H,gx1_.gpu_diff(),gy1_.gpu_diff(),d2_tmp_.mutable_gpu_diff());
    
  
     sumflow_diff<Dtype><<<CAFFE_GET_BLOCKS(verti_loss_.count()), CAFFE_CUDA_NUM_THREADS>>>(verti_loss_.count(),W,H,d2_tmp_.gpu_diff(),bottom[0]->mutable_gpu_diff());
    
     caffe_gpu_set(d2_tmp_.count() , Dtype(0), d2_tmp_.mutable_gpu_diff());
 
     computeflow_diff2<Dtype><<<CAFFE_GET_BLOCKS(verti_loss_.count()), CAFFE_CUDA_NUM_THREADS>>>(verti_loss_.count(),W,H,gx2_.gpu_diff(),gy2_.gpu_diff(),d2_tmp_.mutable_gpu_diff());

     sumflow_diff<Dtype><<<CAFFE_GET_BLOCKS(verti_loss_.count()), CAFFE_CUDA_NUM_THREADS>>>(verti_loss_.count(),W,H,d2_tmp_.gpu_diff(),bottom[0]->mutable_gpu_diff());
 
    }   
  
}

INSTANTIATE_LAYER_GPU_FUNCS(SmoothLossLayer);

}  // namespace caffe
