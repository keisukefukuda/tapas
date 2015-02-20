#include <iostream>
#include <iomanip>

#include "tapas_exafmm.h"

#define ODDEVEN(n) ((((n) & 1) == 1) ? -1 : 1)
#define IPOW2N(n) ((n >= 0) ? 1 : ODDEVEN(n))

// for debugging
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <fstream>
#include <iomanip>

#ifdef EXAFMM_TAPAS_MPI
#include <mpi.h>
#endif

namespace {

const complex_t I(0.,1.);                                       // Imaginary
                                                                // unit
template <int DIM, class FP> inline
tapas::Vec<DIM, FP> &asn(tapas::Vec<DIM, FP> &dst, const vec<DIM, FP> &src) {
  for (int i = 0; i < DIM; ++i) {
    dst[i] = src[i];
  }
  return dst;
}
template <int DIM, class FP> inline
vec<DIM, FP> &asn(vec<DIM, FP> &dst, const tapas::Vec<DIM, FP> &src) {
  for (int i = 0; i < DIM; ++i) {
    dst[i] = src[i];
  }
  return dst;
}

template <int DIM, class FP> inline
vec<DIM, FP> tovec(const tapas::Vec<DIM, FP> &src) {
  vec<DIM, FP> dst;
  for (int i = 0; i < DIM; ++i) {
    dst[i] = src[i];
  }
  return dst;
}

//! Get r,theta,phi from x,y,z
void cart2sph(real_t & r, real_t & theta, real_t & phi, vec3 dX) {
  r = sqrt(norm(dX));                                           // r = sqrt(x^2 + y^2 + z^2)
  theta = r == 0 ? 0 : acos(dX[2] / r);                         // theta = acos(z / r)
  phi = atan2(dX[1], dX[0]);                                    // phi = atan(y / x)
}

//! Spherical to cartesian coordinates
template<typename T>
void sph2cart(real_t r, real_t theta, real_t phi, T spherical, T & cartesian) {
  cartesian[0] = sin(theta) * cos(phi) * spherical[0]           // x component (not x itself)
    + cos(theta) * cos(phi) / r * spherical[1]
    - sin(phi) / r / sin(theta) * spherical[2];
  cartesian[1] = sin(theta) * sin(phi) * spherical[0]           // y component (not y itself)
    + cos(theta) * sin(phi) / r * spherical[1]
    + cos(phi) / r / sin(theta) * spherical[2];
  cartesian[2] = cos(theta) * spherical[0]                      // z component (not z itself)
    - sin(theta) / r * spherical[1];
}

//! Evaluate solid harmonics \f$ r^n Y_{n}^{m} \f$
void evalMultipole(real_t rho, real_t alpha, real_t beta, complex_t * Ynm, complex_t * YnmTheta) {
  real_t x = std::cos(alpha);                                   // x = cos(alpha)
  real_t y = std::sin(alpha);                                   // y = sin(alpha)
  real_t fact = 1;                                              // Initialize 2 * m + 1
  real_t pn = 1;                                                // Initialize Legendre polynomial Pn
  real_t rhom = 1;                                              // Initialize rho^m
  complex_t ei = std::exp(I * beta);                            // exp(i * beta)
  complex_t eim = 1.0;                                          // Initialize exp(i * m * beta)
  for (int m=0; m<P; m++) {                                     // Loop over m in Ynm
    real_t p = pn;                                              //  Associated Legendre polynomial Pnm
    int npn = m * m + 2 * m;                                    //  Index of Ynm for m > 0
    int nmn = m * m;                                            //  Index of Ynm for m < 0
    Ynm[npn] = rhom * p * eim;                                  //  rho^m * Ynm for m > 0
    Ynm[nmn] = std::conj(Ynm[npn]);                             //  Use conjugate relation for m < 0
    real_t p1 = p;                                              //  Pnm-1
    p = x * (2 * m + 1) * p1;                                   //  Pnm using recurrence relation
    YnmTheta[npn] = rhom * (p - (m + 1) * x * p1) / y * eim;    //  theta derivative of r^n * Ynm
    rhom *= rho;                                                //  rho^m
    real_t rhon = rhom;                                         //  rho^n
    for (int n=m+1; n<P; n++) {                                 //  Loop over n in Ynm
      int npm = n * n + n + m;                                  //   Index of Ynm for m > 0
      int nmm = n * n + n - m;                                  //   Index of Ynm for m < 0
      rhon /= -(n + m);                                         //   Update factorial
      Ynm[npm] = rhon * p * eim;                                //   rho^n * Ynm
      Ynm[nmm] = std::conj(Ynm[npm]);                           //   Use conjugate relation for m < 0
      real_t p2 = p1;                                           //   Pnm-2
      p1 = p;                                                   //   Pnm-1
      p = (x * (2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1);  //   Pnm using recurrence relation
      YnmTheta[npm] = rhon * ((n - m + 1) * p - (n + 1) * x * p1) / y * eim;// theta derivative
      rhon *= rho;                                              //   Update rho^n
    }                                                           //  End loop over n in Ynm
    rhom /= -(2 * m + 2) * (2 * m + 1);                         //  Update factorial
    pn = -pn * fact * y;                                        //  Pn
    fact += 2;                                                  //  2 * m + 1
    eim *= ei;                                                  //  Update exp(i * m * beta)
  }                                                             // End loop over m in Ynm
}

//! Evaluate singular harmonics \f$ r^{-n-1} Y_n^m \f$
void evalLocal(real_t rho, real_t alpha, real_t beta, complex_t * Ynm) {
  real_t x = std::cos(alpha);                                   // x = cos(alpha)
  real_t y = std::sin(alpha);                                   // y = sin(alpha)
  real_t fact = 1;                                              // Initialize 2 * m + 1
  real_t pn = 1;                                                // Initialize Legendre polynomial Pn
  real_t invR = -1.0 / rho;                                     // - 1 / rho
  real_t rhom = -invR;                                          // Initialize rho^(-m-1)
  complex_t ei = std::exp(I * beta);                            // exp(i * beta)
  complex_t eim = 1.0;                                          // Initialize exp(i * m * beta)
  for (int m=0; m<P; m++) {                                     // Loop over m in Ynm
    real_t p = pn;                                              //  Associated Legendre polynomial Pnm
    int npn = m * m + 2 * m;                                    //  Index of Ynm for m > 0
    int nmn = m * m;                                            //  Index of Ynm for m < 0
    Ynm[npn] = rhom * p * eim;                                  //  rho^(-m-1) * Ynm for m > 0
    Ynm[nmn] = std::conj(Ynm[npn]);                             //  Use conjugate relation for m < 0
    real_t p1 = p;                                              //  Pnm-1
    p = x * (2 * m + 1) * p1;                                   //  Pnm using recurrence relation
    rhom *= invR;                                               //  rho^(-m-1)
    real_t rhon = rhom;                                         //  rho^(-n-1)
    for (int n=m+1; n<P; n++) {                                 //  Loop over n in Ynm
      int npm = n * n + n + m;                                  //   Index of Ynm for m > 0
      int nmm = n * n + n - m;                                  //   Index of Ynm for m < 0
      Ynm[npm] = rhon * p * eim;                                //   rho^n * Ynm for m > 0
      Ynm[nmm] = std::conj(Ynm[npm]);                           //   Use conjugate relation for m < 0
      real_t p2 = p1;                                           //   Pnm-2
      p1 = p;                                                   //   Pnm-1
      p = (x * (2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1);  //   Pnm using recurrence relation
      rhon *= invR * (n - m + 1);                               //   rho^(-n-1)
    }                                                           //  End loop over n in Ynm
    pn = -pn * fact * y;                                        //  Pn
    fact += 2;                                                  //  2 * m + 1
    eim *= ei;                                                  //  Update exp(i * m * beta)
  }                                                             // End loop over m in Ynm
}
}
void tapas_kernel::P2M(Tapas::Cell &C) {
  complex_t Ynm[P*P], YnmTheta[P*P];
  
  Stderr e("P2M");
  
  for (tapas::index_t i = 0; i < C.nb(); ++i) {
    const Body &B = C.body(i);
    vec3 dX = B.X - tovec(C.center());
    real_t rho, alpha, beta;
    cart2sph(rho, alpha, beta, dX);
    evalMultipole(rho, alpha, beta, Ynm, YnmTheta);
    e.out() << std::setw(10) << tapas::morton_common::SimplifyKey(C.key())
            << " B[" << i << "].SRC=" << B.SRC << std::endl;
    for (int n=0; n<P; n++) {
      for (int m=0; m<=n; m++) {
        int nm  = n * n + n - m;
        int nms = n * (n + 1) / 2 + m;
        C.attr().M[nms] += B.SRC * Ynm[nm];
      }
    }
  }
  e.out() << std::setw(10) << tapas::morton_common::SimplifyKey(C.key())
          << "M=" << C.attr().M
          << std::endl;
}

void tapas_kernel::M2M(Tapas::Cell & C) {
  complex_t Ynm[P*P], YnmTheta[P*P];

  tapas::morton_common::KeyType debug_key = 7905;
  
  if (C.key() % 10000 == debug_key) {
    Stderr e("M2M");
    e.out() << "M2M "
            << tapas::morton_common::SimplifyKey(C.key()) << ", "
            << C.depth() << ", "
            << C.center() << std::endl;
  }
  
  for (int i = 0; i < C.nsubcells(); ++i) {
    Tapas::Cell &Cj=C.subcell(i);
    
    if (C.key() % 10000 == debug_key) {
      Stderr e("M2M");
      e.out() << "Cj=subcell " << i << std::endl;
      e.out() << "Cj.key = " << tapas::morton_common::SimplifyKey(Cj.key()) << std::endl;
      e.out() << "Cj.nb() = " << Cj.nb() << std::endl;
      e.out() << "Cj.center() = " << Cj.center() << std::endl;
      e.out() << "Cj.M = " << Cj.attr().M << std::endl;
#if EXAFMM_TAPAS_MPI
      //e.out() << "Cj.IsLocal() = " << Cj.IsLocal() << std::endl;
#endif
    }
    
    // Skip empty cell
    if (Cj.nb() == 0) continue;
    vec3 dX = tovec(C.center() - Cj.center());
    
    real_t rho, alpha, beta;
    cart2sph(rho, alpha, beta, dX);
    evalMultipole(rho, alpha, beta, Ynm, YnmTheta);

    if (C.key() % 10000 == debug_key) {
      Stderr e("M2M");
      e.out() << "dX=" << dX << std::endl;
      e.out() << "rho=" << rho << std::endl;
      e.out() << "alpha=" << alpha << std::endl;
      e.out() << "Ynm=";
      for (auto &y: Ynm) e.out() << y << ",";
      e.out() << std::endl;
      e.out() << "YnmTheta=";
      for (auto &y: YnmTheta) e.out() << y << ",";
      e.out() << std::endl;
    }

    for (int j=0; j<P; j++) {
      for (int k=0; k<=j; k++) {
        int jks = j * (j + 1) / 2 + k;
        complex_t M = 0;
        for (int n=0; n<=j; n++) {
          for (int m=std::max(-n,-j+k+n); m<=std::min(k-1,n); m++) {
            int jnkms = (j - n) * (j - n + 1) / 2 + k - m;
            int nm    = n * n + n - m;
            M += Cj.attr().M[jnkms] * Ynm[nm] * real_t(IPOW2N(m) * ODDEVEN(n));
          }
          for (int m=k; m<=std::min(n,j+k-n); m++) {
            int jnkms = (j - n) * (j - n + 1) / 2 - k + m;
            int nm    = n * n + n - m;
            M += std::conj(Cj.attr().M[jnkms]) * Ynm[nm] * real_t(ODDEVEN(k+n+m));
          }
        }
        C.attr().M[jks] += M;
      }
    }
  }
  if (C.key() % 10000 == debug_key) {
    Stderr e("M2M");
    for (int i = 0; i < C.nsubcells(); i++) {
      Tapas::Cell &Cj = C.subcell(i);
      e.out() << "C[" << i << "].key = " << tapas::morton_common::SimplifyKey(Cj.key()) << std::endl;
      e.out() << "C[" << i << "].IsLeaf = " << Cj.IsLeaf() << std::endl;
#if EXAFMM_TAPAS_MPI
      //e.out() << "C[" << i << "].Local  = " << Cj.IsLocal() << std::endl;
#endif 
      e.out() << "C[" << i << "].depth  = " << Cj.depth() << std::endl;
      e.out() << "C[" << i << "].center = " << Cj.center() << std::endl;
      e.out() << "C[" << i << "].M = " << Cj.attr().M << std::endl;
    }
    
    e.out() << "M=" << C.attr().M << std::endl;
  }
}

void tapas_kernel::M2L(Tapas::Cell &Ci, Tapas::Cell &Cj, vec3 Xperiodic, bool mutual) {
  complex_t Ynmi[P*P], Ynmj[P*P];
  //vec3 dX = Ci.attr().X - Cj.attr().X - Xperiodic;
  vec3 dX;
  asn(dX, Ci.center() - Cj.center());
  dX -= Xperiodic;
  //std::cerr << "dx: " << dX << std::endl;  
  real_t rho, alpha, beta;
  cart2sph(rho, alpha, beta, dX);
  // std::cerr  << "rho: " << rho << ", alpha: " << alpha << ", beta: " << beta << std::endl;
  evalLocal(rho, alpha, beta, Ynmi);
  for (int i = 0; i < P*P; ++i) {
    // std::cerr << "tapas_kernel::M2L Y: " << Ynmi[i] << std::endl;
  }
  if (mutual) evalLocal(rho, alpha+M_PI, beta, Ynmj);

  for (int j=0; j<P; j++) {
#if MASS
    real_t Cnm = std::real(Ci->M[0] * Cj->M[0]) * ODDEVEN(j);
#else
    real_t Cnm = ODDEVEN(j);
#endif
    for (int k=0; k<=j; k++) {
      int jks = j * (j + 1) / 2 + k;
      complex_t Li = 0, Lj = 0;
#if MASS
      int jk = j * j + j - k;
      Li += Cnm * Ynmi[jk];
      if (mutual) Lj += Cnm * Ynmj[jk];
      for (int n=1; n<P-j; n++)
#else
      for (int n=0; n<P-j; n++)
#endif
      {
        for (int m=-n; m<0; m++) {
          int nms  = n * (n + 1) / 2 - m;
          int jnkm = (j + n) * (j + n) + j + n + m - k;
          Li += std::conj(Cj.attr().M[nms]) * Cnm * Ynmi[jnkm];
          //std::cerr << "M: " << Cj.attr().M[nms] << std::conj(Cj.attr().M[nms]) << std::endl;
          //std::cerr << "Y: " << Ynmi[jnkm] << std::endl;
          if (mutual) Lj += std::conj(Ci.attr().M[nms]) * Cnm * Ynmj[jnkm];
        }
        for (int m=0; m<=n; m++) {
          int nms  = n * (n + 1) / 2 + m;
          int jnkm = (j + n) * (j + n) + j + n + m - k;
          real_t Cnm2 = Cnm * ODDEVEN((k-m)*(k<m)+m);
          Li += Cj.attr().M[nms] * Cnm2 * Ynmi[jnkm];
          //std::cerr << "M: " << Cj.attr().M[nms] << std::conj(Cj.attr().M[nms]) << std::endl;
          if (mutual) Lj += Ci.attr().M[nms] * Cnm2 * Ynmj[jnkm];
        }
      }
      Ci.attr().L[jks] += Li;
      //std::cerr << "Li: " << Li << std::endl;
      if (mutual) Cj.attr().L[jks] += Lj;
    }
  }
}


void tapas_kernel::L2P(Tapas::BodyIterator &B) {
  complex_t Ynm[P*P], YnmTheta[P*P];
  const Tapas::Cell &C = B.cell();
  vec3 dX = B->X - tovec(C.center());
  vec3 spherical = 0;
  vec3 cartesian = 0;
  real_t r, theta, phi;
  cart2sph(r, theta, phi, dX);
  evalMultipole(r, theta, phi, Ynm, YnmTheta);
  B.attr() /= B->SRC;
  for (int n=0; n<P; n++) {
    int nm  = n * n + n;
    int nms = n * (n + 1) / 2;
    //B->TRG[0] += std::real(C.attr().L[nms] * Ynm[nm]);
    B.attr()[0] += std::real(C.attr().L[nms] * Ynm[nm]);
    spherical[0] += std::real(C.attr().L[nms] * Ynm[nm]) / r * n;
    spherical[1] += std::real(C.attr().L[nms] * YnmTheta[nm]);
    for( int m=1; m<=n; m++) {
      nm  = n * n + n + m;
      nms = n * (n + 1) / 2 + m;
      //B->TRG[0] += 2 * std::real(C.attr().L[nms] * Ynm[nm]);
      B.attr()[0] += 2 * std::real(C.attr().L[nms] * Ynm[nm]);
      spherical[0] += 2 * std::real(C.attr().L[nms] * Ynm[nm]) / r * n;
      spherical[1] += 2 * std::real(C.attr().L[nms] * YnmTheta[nm]);
      spherical[2] += 2 * std::real(C.attr().L[nms] * Ynm[nm] * I) * m;
    }
  }
  sph2cart(r, theta, phi, spherical, cartesian);
  B.attr()[1] += cartesian[0];
  //B->TRG[2] += cartesian[1];
  B.attr()[2] += cartesian[1];
  //B->TRG[3] += cartesian[2];
  B.attr()[3] += cartesian[2]; 
}

void tapas_kernel::L2L(Tapas::Cell &C) {
  assert(0);
  std::cerr << "***********L2L" << std::endl;
  complex_t Ynm[P*P], YnmTheta[P*P];
  const Tapas::Cell &Cj = C.parent();
  vec3 dX = tovec(C.center() - Cj.center());
  real_t rho, alpha, beta;

  Stderr e("L2L");
  e.out() << "C.key = " << tapas::morton_common::SimplifyKey(Cj.key()) << std::endl;
  e.out() << "C.IsLeaf = " << Cj.IsLeaf() << std::endl;
#if EXAFMM_TAPAS_MPI
  //e.out() << "C[" << i << "].Local  = " << Cj.IsLocal() << std::endl;
#endif 
  e.out() << "C.depth  = " << Cj.depth() << std::endl;
  e.out() << "C.center = " << Cj.center() << std::endl;
  
  cart2sph(rho, alpha, beta, dX);
  evalMultipole(rho, alpha, beta, Ynm, YnmTheta);
#if MASS
  C.attr().L /= C.attr().M[0];
#endif
  for (int j=0; j<P; j++) {
    for (int k=0; k<=j; k++) {
      int jks = j * (j + 1) / 2 + k;
      complex_t L = 0;
      for (int n=j; n<P; n++) {
        for (int m=j+k-n; m<0; m++) {
          int jnkm = (n - j) * (n - j) + n - j + m - k;
          int nms  = n * (n + 1) / 2 - m;
          L += std::conj(Cj.attr().L[nms]) * Ynm[jnkm] * real_t(ODDEVEN(k));
        }
        for (int m=0; m<=n; m++) {
          if( n-j >= abs(m-k) ) {
            int jnkm = (n - j) * (n - j) + n - j + m - k;
            int nms  = n * (n + 1) / 2 + m;
            L += Cj.attr().L[nms] * Ynm[jnkm] * real_t(ODDEVEN((m-k)*(m<k)));
          }
        }
      }
      C.attr().L[jks] += L;
    }
  }

  e.out() << "Cj.key = " << tapas::morton_common::SimplifyKey(Cj.key()) << std::endl;
  e.out() << "Cj.IsLeaf = " << Cj.IsLeaf() << std::endl;
#if EXAFMM_TAPAS_MPI
  //e.out() << "C[" << i << "].Local  = " << Cj.IsLocal() << std::endl;
#endif 
  e.out() << "Cj.depth  = " << Cj.depth() << std::endl;
  e.out() << "Cj.center = " << Cj.center() << std::endl;
  e.out() << "Cj.L = " << Cj.attr().L << std::endl;
  e.out() << "C.L=" << C.attr().L << std::endl;
}
