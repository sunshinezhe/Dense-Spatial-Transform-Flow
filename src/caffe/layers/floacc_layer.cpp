#include <functional>
#include <utility>
#include <vector>
#include <cmath>
#include "caffe/layers/floacc_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FloaccLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
 
}

template <typename Dtype>
void FloaccLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

   N=bottom[0]->shape(0);
   C=bottom[0]->shape(1);
   H=bottom[0]->shape(2);
   W=bottom[0]->shape(3);

  error_.Reshape(N,1,H,W);
  mask_.Reshape(N,C,H,W);
  sign_.Reshape(N,1,H,W);
  caffe_set(sign_.count(), Dtype(1), sign_.mutable_cpu_data());


  vector<int> top1_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top1_shape);

  vector<int> top2_shape(1);
  top2_shape[0]=4;
  top[1]->Reshape(top2_shape);

  
}

template <typename Dtype>
void FloaccLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  CHECK(false) << "FloaccLayer cannot do backward.";
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(FloaccLayer);
REGISTER_LAYER_CLASS(Floacc);

}  // namespace caffe
