#ifndef CAFFE_DATA_LOSS_HPP_
#define CAFFE_DATA_LOSS_HPP_

#include <string>
#include <utility>
#include <vector>


#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"


namespace caffe {



//added by Ren.Zhe 
// Computes data loss! 
//Forward declare
template <typename Dtype> class ConvolutionLayer;
template <typename Dtype> class EltwiseLayer;
template <typename Dtype> class SpatialTransformerLayer;
template <typename Dtype> class PowerLayer;


template <typename Dtype>
class DataLossLayer : public LossLayer<Dtype> {
 public:
  explicit DataLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), sign_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);    
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DataLoss"; }

  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }
  virtual inline int ExactNumBottomBlobs() const { return 3; }  
  

 protected:
  /// @copydoc DataLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> sign_, mask_, plateau_l2_;
  float scale_;
  Dtype normalize_coeff_;
  
  // Extra layers to do the dirty work using already implemented stuff
  shared_ptr<EltwiseLayer<Dtype> > diff_layer_;
  Blob<Dtype> diff_;
  vector<Blob<Dtype>*> diff_top_vec_;
  shared_ptr<PowerLayer<Dtype> > square_layer_;
  Blob<Dtype> square_output_;
  vector<Blob<Dtype>*> square_top_vec_;
  shared_ptr<ConvolutionLayer<Dtype> > sum_layer_;
  Blob<Dtype> sum_output_;
  vector<Blob<Dtype>*> sum_top_vec_;
  shared_ptr<PowerLayer<Dtype> > sqrt_layer_;
  Blob<Dtype> sqrt_output_;
  vector<Blob<Dtype>*> sqrt_top_vec_;

  //added by Ren.Zhe
  vector<Blob<Dtype>*> diff_bottom_vec_;
  
  shared_ptr<SpatialTransformerLayer<Dtype> > stn_layer_;
  Blob<Dtype> stn_output_;
  vector<Blob<Dtype>*> stn_top_vec_;
  vector<Blob<Dtype>*> stn_bottom_vec_;
};










}  // namespace caffe

#endif  // CAFFE_DATA_LOSS_HPP_
