/* allpairs-tuple.h (2015-11-11) -*-Coding: us-ascii-unix;-*- */
/* Copyright (C) 2015-2015 RIKEN AICS */

/* Tapas All-Pairs */

#pragma once

#define BR0_ {
#define BR1_ }

namespace tapas BR0_

template <class T0, class T1>
struct tuple;

template <size_t I, class T>
struct tuple_element_;

template <class T0, class T1>
struct tuple_element_<0, tuple<T0, T1>> {
    typedef T0 type;
};

template <class T0, class T1>
struct tuple_element_<1, tuple<T0, T1>> {
    typedef T1 type;
};

template <class T0, class T1>
struct tuple {
    T0& _0;
    T1& _1;

    /*tuple() {}*/

    tuple(const tuple&) = default;

    tuple(tuple&&) = default;

    __host__ __device__ inline
    tuple(T0& u0, T1& u1) : _0 (u0), _1 (u1) {}

    /*tuple(T0&& u0, T1&& u1) : _0 (u0), _1 (u1) {}*/

    __host__ __device__ inline
    tuple& operator=(const tuple& u) {
	this->_0 = u._0;
	this->_1 = u._1;
	return *this;
    }

    __host__ __device__ inline
    tuple& operator=(tuple&& u) noexcept {
	this->_0 = u._0;
	this->_1 = u._1;
	return *this;
    }
};

template <size_t I, class U0, class U1>
struct tuple_get_ {
    __host__ __device__ inline
    static const tuple_element_<I, tuple<U0, U1>>&
	get(const tuple<U0, U1>& u) noexcept;
};

template <class U0, class U1>
struct tuple_get_<0, U0, U1> {
    __host__ __device__ inline
    static const U0& get(const tuple<U0, U1>& u) {
	return u._0;
    }
};

template <class U0, class U1>
struct tuple_get_<1, U0, U1> {
    __host__ __device__ inline
    static const U1& get(const tuple<U0, U1>& u) {
	return u._1;
    }
};

template <size_t I, class U0, class U1>
__host__ __device__ inline
const typename tuple_element_<I, tuple<U0, U1>>::type&
    get(const tuple<U0, U1>& u) noexcept {
    return tuple_get_<I, U0, U1>::get(u);
}

template <class T>
struct vector {
    typedef T value_type;

    T* v;
    size_t sz;

    __host__ __device__ inline
    vector<T>(T* v_, size_t sz_) : v (v_), sz (sz_) {}

    __host__ __device__ inline
    T& operator[](const size_t i) {
	return v[i];
    }

    __host__ __device__ inline
    T* data() {
	return v;
    }

    __host__ __device__ inline
    size_t size() {
	return sz;
    }
};

template <class T0, class T1>
struct zipped {
    typedef tuple<T0, T1> value_type;

    T0* _0;
    T1* _1;
    size_t sz_;

    __host__ __device__ inline
    zipped<T0, T1>(T0* u0, T1* u1) : _0 (u0), _1 (u1), sz_ (0) {}

    __host__ __device__ inline
    tuple<T0, T1> operator[](const size_t i) {
	return tuple<T0, T1>(_0[i], _1[i]);
    }
};

BR1_

#undef BR0_
#undef BR1_

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
