#ifndef CAFFE_DYNAMIC_BILINEAR_HPP_
#define CAFFE_DYNAMIC_BILINEAR_HPP_

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class DynamicBilinearLayer : public Layer<Dtype> {

public:
	explicit DynamicBilinearLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {
	      
	      global_debug = false; 	      
      }
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "DynamicBilinear"; }

        virtual inline int MinBottomBlobs() const { return 1; }
        virtual inline int MaxBottomBlobs() const { return 2; }
	
	virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

private:
	inline Dtype abs(Dtype x) {
		if(x < 0) return -x; return x;
	}
	inline Dtype max(Dtype x, Dtype y) {
		if(x < y) return y; return x;
	}


	int output_H_;
	int output_W_;

	int N, C, H, W;

	bool global_debug;
	



};

}  // namespace caffe

#endif  // CAFFE_DYNAMIC_BILINEAR_HPP_
