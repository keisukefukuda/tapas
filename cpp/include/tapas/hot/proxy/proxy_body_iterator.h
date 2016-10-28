#ifndef TAPAS_HOT_PROXY_PROXY_BODY_ITERATOR_H_
#define TAPAS_HOT_PROXY_PROXY_BODY_ITERATOR_H_

#include "tapas/hot/proxy/proxy_mapper.h"

namespace tapas {
namespace hot {
namespace proxy {

/**
 * \brief Iterator class for ProxyBody.
 *
 * Iterators are to be removed by design refactoring. 
 * Iterators imply "iterating for loop" but we don't want users to use for loops over Cells/Bodies.
 */
template<class PROXY_CELL>
class ProxyBodyIterator {
 public:
  using TSP = typename PROXY_CELL::TSP;
  using CellType = PROXY_CELL;
  using CellAttr = ProxyBodyAttr<PROXY_CELL>;
  using Body = typename TSP::Body;
  using value_type = ProxyBodyIterator<PROXY_CELL>;
  //using Mapper = typename CellType::Mapper;
  using Mapper = ProxyMapper<PROXY_CELL>;

 private:
  PROXY_CELL *c_;
  index_t idx_;
  Mapper mapper_;

 public:
  static const constexpr int kThreadSpawnThreshold = 100;

  ProxyBodyIterator(PROXY_CELL *c) : c_(c), idx_(0), mapper_() { }

  ProxyBodyIterator &operator*() {
    return *this;
  }

  constexpr bool SpawnTask() const { return false; }

  Mapper &mapper() { return mapper_; }
  const Mapper &mapper() const { return c_->mapper(); }

  inline int index() const { return idx_; }

  PROXY_CELL &cell() const {
    return *c_;
  }

  const ProxyBodyIterator &operator*() const {
    return *this;
  }

  bool operator==(const ProxyBodyIterator &x) const {
    return *c_ == *(x.c_) && idx_ == x.idx_;
  }

  bool operator<(const ProxyBodyIterator &x) const {
    if (*c_ == *x.c_) {
      return idx_ < x.idx_;
    } else {
      return *c_ < *x.c_;
    }
  }

  template<class T>
  bool operator==(const T&) const {
    return false;
  }

  /**
   * \fn bool ProxyBodyIterator::operator!=(const ProxyBodyIterator &x) const
   */
  bool operator!=(const ProxyBodyIterator &x) const {
    return !(*this == x);
  }

  /**
   * \fn ProxyBody &ProxyBodyIterator::operator++()
   */
  const ProxyBody<PROXY_CELL> &operator++() {
    return c_->body(++idx_);
  }

  /**
   * \fn ProxyBody &ProxyBodyIterator::operator++(int)
   */
  const ProxyBody<PROXY_CELL> &operator++(int) {
    return c_->body(idx_++);
  }

  ProxyBodyIterator operator+(int i) {
    ProxyBodyIterator ret = *this;
    ret.idx_ += i;
    TAPAS_ASSERT(ret.idx_ < size());
    return ret;
  }

  /**
   * \fn void ProxyBodyIterator::rewind(int idx)
   */
  void rewind(int idx) {
    idx_ = idx;
  }

  /**
   * \fn bool ProxyBodyIterator::AllowMutualInteraction(const ProxyBodyIterator &x) const;
   */
  bool AllowMutualInteraction(const ProxyBodyIterator &x) const {
    return *c_ == *(x.c_);
  }

  index_t size() const {
    return c_->nb();
  }

  bool IsLocal() const {
    return c_->IsLocal();
  }

  ProxyBodyIterator &operator+=(int n) {
    idx_ += n;
    TAPAS_ASSERT(idx_ < c_->RealCell()->nb());
    return *this;
  }

  // Returns a const pointer to (real) Body for read-only use.
  const Body *operator->() const {
    return reinterpret_cast<const Body*>(&(c_->body(idx_)));
  }

  const ProxyBodyAttr<PROXY_CELL> &attr() const {
    return c_->body_attr(idx_);
  }

}; // class ProxyBodyIterator

} // namespace proxy
} // namespace hot
} // namespace tapas

#endif // TAPAS_HOT_PROXY_PROXY_BODY_ITERATOR_H_
