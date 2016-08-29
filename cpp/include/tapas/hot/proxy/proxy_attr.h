#ifndef TAPAS_HOT_PROXY_PROXY_ATTR_H_
#define TAPAS_HOT_PROXY_PROXY_ATTR_H_

namespace tapas {
namespace hot {
namespace proxy {

template<class TSP> class ProxyCell;

/**
 * \brief Proxy class for user-specified CellAttr.
 */
template<class TSP>
class ProxyAttr : public TSP::CellAttr {
  friend ProxyCell<TSP>;

  using CellAttr = typename TSP::CellAttr;
    
 protected:
  ProxyAttr &operator=(const ProxyAttr &rhs) {
    this->CellAttr::operator=(rhs);
    cell_ = rhs.cell_;
    return *this;
  }
    
 public:
  ProxyAttr(ProxyCell<TSP> *cell) : CellAttr(), cell_(cell) { }
  ProxyAttr(ProxyCell<TSP> *cell, CellAttr &rhs) : CellAttr(rhs), cell_(cell) { }

  //ProxyAttr(const ProxyAttr &) = delete;
    
  ProxyCell<TSP> &cell() const { return *cell_; }

  inline void operator=(const CellAttr&) const {
    //std::cout << "ProxyAttr::operator=() is called for cell [" << cell_->key() << "]" << std::endl;
    cell_->MarkModified();
    //return *this;
  }

  template<class T>
  inline void operator=(const CellAttr&) const {
    //std::cout << "ProxyAttr::operator=() const is called for cell [" << cell_->key() << "]" << std::endl;
    cell_->MarkModified();
    //return *this;
  }

 private:
  ProxyCell<TSP> *cell_;
}; // class ProxyAttr


} // namespace proxy
} // namespace hot
} // namespace proxy

#endif // TAPAS_HOT_PROXY_PROXY_ATTR_H_
