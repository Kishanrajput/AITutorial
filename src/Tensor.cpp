//
// Kishansingh Rajput - JLab
//

#include "../include/Tensor.h"
//#include<stdio>

#include <utility>
#include <utility>
#include <cstddef>
#include <memory>
#include <type_traits>

namespace std {
    template<class T> struct _Unique_if {
        typedef unique_ptr<T> _Single_object;
    };

    template<class T> struct _Unique_if<T[]> {
        typedef unique_ptr<T[]> _Unknown_bound;
    };

    template<class T, size_t N> struct _Unique_if<T[N]> {
        typedef void _Known_bound;
    };

    template<class T, class... Args>
    typename _Unique_if<T>::_Single_object
    make_unique(Args&&... args) {
        return unique_ptr<T>(new T(std::forward<Args>(args)...));
    }

    template<class T>
    typename _Unique_if<T>::_Unknown_bound
    make_unique(size_t n) {
        typedef typename remove_extent<T>::type U;
        return unique_ptr<T>(new U[n]());
    }

    template<class T, class... Args>
    typename _Unique_if<T>::_Known_bound
    make_unique(Args&&...) = delete;
}

Tensor::Tensor(const Model& model, const std::string& operation, int shape1, int shape2) {

    // Get operation by the name
    this->op.oper = TF_GraphOperationByName(model.graph, operation.c_str());
    this->op.index = 0;
    //std::cout<<"00000000000000000000000000000000002"<<std::endl;

    // Operation did not exists
    error_check(this->op.oper != nullptr, "No operation named \"" + operation + "\" exists" );

    //std::cout<<"00000000000000000000000000000000003"<<std::endl;
    // DIMENSIONS

    // Get number of dimensions
    int n_dims = TF_GraphGetTensorNumDims(model.graph, this->op, model.status);

    //std::cout<<"00000000000000000000000000000000004"<<std::endl;
    // DataType
    this->type = TF_OperationOutputType(this->op);

    //std::cout<<"00000000000000000000000000000000005"<<std::endl;
    // If is not a scalar
    if (n_dims > 0) {
        // Get dimensions
        auto *dims = new int64_t[n_dims];
        TF_GraphGetTensorShape(model.graph, this->op, dims, n_dims, model.status);
        // Check error on Model Status
        model.status_check(true);

        //this->shape = std::vector<int64_t>(dims, dims + n_dims);

        this->shape.push_back(shape1);
        this->shape.push_back(shape2);
        // Only one dimension can be unknown using this constructor
        // error_check(std::count(this->shape.begin(), this->shape.end(), -1) <= 1, "At most one dimension can be unknown");

        delete[] dims;
    }

    //std::cout<<"00000000000000000000000000000000006"<<std::endl;
    this->flag = 0;
    this->val = nullptr;
    this->data = nullptr;
    //std::cout<<"Shape of Tensor "<<this->shape<<std::endl;
}

Tensor::~Tensor() {
    this->clean();
}



void Tensor::clean() {
    if (this->flag == 1) {
        TF_DeleteTensor(this->val);
        this->flag = 0;
    }
    this->data = nullptr;
}

void  Tensor::error_check(bool condition, const std::string &error) {
    if (!condition) {
        this->flag = -1;
        throw std::runtime_error(error);
    }
}

template<typename T>
void Tensor::set_data(std::vector<T> new_data) {

    //Non empty tensor
    if (this->flag == 1) {
        TF_DeleteTensor(this->val);
        this->flag = 0;
    }

    // -- Print Shape of the Tensor --------------------------------------
    // std::cout<<"Shape of Tensor ";
    // for(int i=0; i < this->shape.size(); i++)
    // {
    //     std::cout << this->shape.at(i) << ' ';     
    // }
    // std::cout<<std::endl;
    // --------------------------------------------------------------------
   

    // Check Tensor is valid
    this->error_check(this->flag != -1, "Tensor is not valid");

    // Check type
    this->error_check(deduce_type<T>() == this->type, "Provided type is different from Tensor expected type");

    // Dimensions must be known
    this->error_check(!this->shape.empty(), "Shape of the input Tensor is not known, please provide a shape");

    // At most one dimension can be unknown
    this->error_check(std::count(this->shape.begin(), this->shape.end(), -1) >= -1, "At most one dimension can be unknown, please provide a shape");

    // Check number of elements
    auto exp_size = std::abs(std::accumulate(this->shape.begin(), this->shape.end(), 1, std::multiplies<int64_t>()));

    this->error_check(new_data.size() % exp_size == 0, "Expected and provided number of elements do not match");

    // Deallocator
    auto d = [](void* ddata, size_t, void*) {free(static_cast<T*>(ddata));};


    // Calculate actual shape of unknown dimensions
    this->actual_shape = std::make_unique<decltype(actual_shape)::element_type>(shape.begin(), shape.end());
    std::replace_if (actual_shape->begin(), actual_shape->end(), [](int64_t r) {return r==-1;}, new_data.size()/exp_size);

    // Saves data on class
    this->data = malloc(sizeof(T) * new_data.size());
    memcpy(this->data, new_data.data(), sizeof(T) * new_data.size());

    this->val = TF_NewTensor(this->type, actual_shape->data(), actual_shape->size(), this->data, sizeof(T) * new_data.size(), d, nullptr);


    this->error_check(this->val != nullptr, "An error occurred allocating the Tensor memory");

    this->flag = 1;
}

template<typename T> void Tensor::set_data(std::vector<T> new_data, const std::vector<int64_t>& new_shape) {

    this->error_check(this->shape.empty() || this->shape.size() == new_shape.size(), "Provided shape has different number of dimensions");
    auto old_shape = this->shape;

    this->shape = new_shape;
    this->set_data(new_data);

    this->shape = old_shape;
}

template<typename T>
std::vector<T> Tensor::get_data() {
    
    //std::cout<<"Inside get_data of Tensor "<<std::endl;
    
    // Check Tensor is valid
    this->error_check(this->flag != -1, "Tensor is not valid");
    //std::cout<<"Inside get_data of Tensor 1"<<std::endl;
    
    // Check type
    //this->error_check(deduce_type<T>() == this->type, "Expected return type is different from Tensor type");
    //std::cout<<"Inside get_data of Tensor 2"<<std::endl;
    
    // Tensor is not empty
    this->error_check(this->flag != 0, "Tensor is empty");
    //std::cout<<"Inside get_data of Tensor 3"<<std::endl;

    // Check tensor data is not empty
    auto raw_data = TF_TensorData(this->val);
    this->error_check(raw_data != nullptr, "Tensor data is empty");

    //std::cout<<"Here at tensor0000"<<std::endl;
    size_t size = TF_TensorByteSize(this->val) / TF_DataTypeSize(TF_TensorType(this->val));
    //std::cout<<"Here at tensor1111"<<std::endl;
    // Convert to correct type
    const auto T_data = static_cast<T*>(raw_data);
    //std::cout<<"Here at tensor2222"<<std::endl;

    return std::vector<T>(T_data, T_data + size);
}

std::vector<int64_t> Tensor::get_shape() {
	return shape;
}

template<typename T>
TF_DataType Tensor::deduce_type() {
    if (std::is_same<T, float>::value)
        return TF_FLOAT;
    if (std::is_same<T, double>::value)
        return TF_DOUBLE;
    if (std::is_same<T, int32_t >::value)
        return TF_INT32;
    if (std::is_same<T, uint8_t>::value)
        return TF_UINT8;
    if (std::is_same<T, int16_t>::value)
        return TF_INT16;
    if (std::is_same<T, int8_t>::value)
        return TF_INT8;
    if (std::is_same<T, int64_t>::value)
        return TF_INT64;
//    if constexpr (std::is_same<T, bool>::value)
//        return TF_BOOL;
    if (std::is_same<T, uint16_t>::value)
        return TF_UINT16;
    if (std::is_same<T, uint32_t>::value)
        return TF_UINT32;
    if (std::is_same<T, uint64_t>::value)
        return TF_UINT64;

    throw std::runtime_error{"Could not deduce type!"};
}

void Tensor::deduce_shape() {
    // Get number of dimensions
    int n_dims = TF_NumDims(this->val);

    // If is not a scalar
    if (n_dims > 0) {
        // Get dimensions
        this->shape = std::vector<int64_t>(n_dims, -1);
        for (int i=0; i<n_dims; i++) {
            this->shape[i] = TF_Dim(this->val, i);
        }
    }
}


// VALID deduce_type TEMPLATES
template TF_DataType Tensor::deduce_type<float>();
template TF_DataType Tensor::deduce_type<double>();
//template TF_DataType Tensor::deduce_type<bool>();
template TF_DataType Tensor::deduce_type<int8_t>();
template TF_DataType Tensor::deduce_type<int16_t>();
template TF_DataType Tensor::deduce_type<int32_t>();
template TF_DataType Tensor::deduce_type<int64_t>();
template TF_DataType Tensor::deduce_type<uint8_t>();
template TF_DataType Tensor::deduce_type<uint16_t>();
template TF_DataType Tensor::deduce_type<uint32_t>();
template TF_DataType Tensor::deduce_type<uint64_t>();

// VALID get_data TEMPLATES
template std::vector<float> Tensor::get_data<float>();
template std::vector<double> Tensor::get_data<double>();
template std::vector<bool> Tensor::get_data<bool>();
template std::vector<int8_t> Tensor::get_data<int8_t>();
template std::vector<int16_t> Tensor::get_data<int16_t>();
template std::vector<int32_t> Tensor::get_data<int32_t>();
template std::vector<int64_t> Tensor::get_data<int64_t>();
template std::vector<uint8_t> Tensor::get_data<uint8_t>();
template std::vector<uint16_t> Tensor::get_data<uint16_t>();
template std::vector<uint32_t> Tensor::get_data<uint32_t>();
template std::vector<uint64_t> Tensor::get_data<uint64_t>();

// VALID set_data TEMPLATES
template void Tensor::set_data<float>(std::vector<float> new_data);
template void Tensor::set_data<double>(std::vector<double> new_data);
//template void Tensor::set_data<bool>(std::vector<bool> new_data);
template void Tensor::set_data<int8_t>(std::vector<int8_t> new_data);
template void Tensor::set_data<int16_t>(std::vector<int16_t> new_data);
template void Tensor::set_data<int32_t>(std::vector<int32_t> new_data);
template void Tensor::set_data<int64_t>(std::vector<int64_t> new_data);
template void Tensor::set_data<uint8_t>(std::vector<uint8_t> new_data);
template void Tensor::set_data<uint16_t>(std::vector<uint16_t> new_data);
template void Tensor::set_data<uint32_t>(std::vector<uint32_t> new_data);
template void Tensor::set_data<uint64_t>(std::vector<uint64_t> new_data);

// VALID set_data TEMPLATES
template void Tensor::set_data<float>(std::vector<float> new_data, const std::vector<int64_t>& new_shape);
template void Tensor::set_data<double>(std::vector<double> new_data, const std::vector<int64_t>& new_shape);
//template void Tensor::set_data<bool>(std::vector<bool> new_data, const std::vector<int64_t>& new_shape);
template void Tensor::set_data<int8_t>(std::vector<int8_t> new_data, const std::vector<int64_t>& new_shape);
template void Tensor::set_data<int16_t>(std::vector<int16_t> new_data, const std::vector<int64_t>& new_shape);
template void Tensor::set_data<int32_t>(std::vector<int32_t> new_data, const std::vector<int64_t>& new_shape);
template void Tensor::set_data<int64_t>(std::vector<int64_t> new_data, const std::vector<int64_t>& new_shape);
template void Tensor::set_data<uint8_t>(std::vector<uint8_t> new_data, const std::vector<int64_t>& new_shape);
template void Tensor::set_data<uint16_t>(std::vector<uint16_t> new_data, const std::vector<int64_t>& new_shape);
template void Tensor::set_data<uint32_t>(std::vector<uint32_t> new_data, const std::vector<int64_t>& new_shape);
template void Tensor::set_data<uint64_t>(std::vector<uint64_t> new_data, const std::vector<int64_t>& new_shape);
