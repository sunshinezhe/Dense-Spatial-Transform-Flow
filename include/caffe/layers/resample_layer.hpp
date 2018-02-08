#ifndef CAFFE_RESAMPLE_LAYER_HPP
#define CAFFE_RESAMPLE_LAYER_HPP


#include <string>
#include <utility>
#include <vector>


#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/common.hpp"


namespace caffe {

/**
 * @brief The ResampleLayer, resamples a feature blob to a smaller or larger size
 *        using different interpolation methods
 */
template <typename Dtype>
class ResampleLayer : public Layer<Dtype> {
 public:
  explicit ResampleLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline bool AllowBackward() const { LOG(WARNING) << "ResampleLayer does not do backward."; return false; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};


}  // namespace caffe

#endif  // CAFFE_RESAMPLE_LAYER_HPP_
