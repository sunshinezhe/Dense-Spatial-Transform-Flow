#ifndef CAFFE_MEAN_LAYER_HPP
#define CAFFE_MEAN_LAYER_HPP

#include <string>
#include <utility>
#include <vector>


#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"



namespace caffe {

/**
 * @brief Can normalize data by subtracting mean and scaling
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class MeanLayer : public Layer<Dtype> {
 public:
  explicit MeanLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~MeanLayer() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "Mean"; }

  virtual inline bool AllowBackward() const { LOG(WARNING) << "gradienttLayer does not do backward."; return false; }

 protected:

  Blob<Dtype> mean_;
  int mean_num;
  float scale;
  

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)  { LOG(FATAL) << "MeanLayer cannot do backward."; return; }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)  { LOG(FATAL) << "MeanLayer cannot do backward."; return; }
 };



}  // namespace caffe

#endif  // CAFFE_MEAN_LAYER_HPP_
