#ifndef CAFFE_SMOOTH_LOSS_HPP_
#define CAFFE_SMOOTH_LOSS_HPP_

#include <string>
#include <utility>
#include <vector>


#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"


namespace caffe {



template <typename Dtype> class PowerLayer;


//added by Zhe.Ren
//compute smooth loss
template <typename Dtype>
class SmoothLossLayer : public LossLayer<Dtype> {
 public:
  explicit SmoothLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), sign_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);    
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SmoothLoss"; }

  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }
  

  virtual inline int ExactNumBottomBlobs() const { return 2; }

 protected:
  /// @copydoc MatchLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype>  plateau1_l2_ , sign_  ;
  float scale_;
  Dtype normalize_coeff_;
  
  // Extra layers to do the dirty work using already implemented stuff
  
  
  vector<Blob<Dtype>*> sqrt1_bottom_vec_;
  shared_ptr<PowerLayer<Dtype> > sqrt1_layer_;
  Blob<Dtype> sqrt1_output_;
  vector<Blob<Dtype>*> sqrt1_top_vec_;

  vector<Blob<Dtype>*> sqrt2_bottom_vec_;
  shared_ptr<PowerLayer<Dtype> > sqrt2_layer_;
  Blob<Dtype> sqrt2_output_;
  vector<Blob<Dtype>*> sqrt2_top_vec_;

//added for smooth loss
  Blob<Dtype> plateau2_l2_;

  int N,C,H,W;
  Blob<Dtype> gx1_;  
  Blob<Dtype> gy1_;
  Blob<Dtype> gx2_;
  Blob<Dtype> gy2_;
 
  Blob<Dtype> horiz_loss_;
  Blob<Dtype> verti_loss_;
  Blob<Dtype> d2_tmp_;
  Blob<Dtype> smoothweight_;
  Blob<Dtype> smoothweight_h_;
  Blob<Dtype> smoothweight_v_;

  Blob<Dtype> image_filter;

  Blob<Dtype> lum_x_;
  Blob<Dtype> lum_y_;
  Blob<Dtype> lum;  
};





}  // namespace caffe

#endif  // CAFFE_SMOOTH_LOSS_HPP_
