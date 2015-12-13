/***************************************************
 * SingleProcess.cpp
 *
 *
 * Cui Jing
 **************************************************/
#include "SingleProcess.h"
#include <vector>
#include <chrono>
#include <random>
#include <iostream>

using std::vector;
using std::cout;
using std::endl;

SingleProcess::SingleProcess(int N, double Ymin, double Ymax, int kmax, int pmax, double Bfield):
    counts(0),
    N(N),
    kmax(kmax),
    pmax(pmax),
    kpmax(kmax + pmax),
    Ymin(Ymin),
    Ymax(Ymax),
    psi(kmax + pmax + 1, vector<double>(N * N * 2)),
    Y(N),
    B(Bfield),
    alpha(kmax + pmax),
    beta(kmax + pmax) {
    
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    
    for(int i = 0; i < N * N; i++) {
        psi[0][2*i] = distribution(generator);
        psi[0][2*i+1] = 0;
    }
    for(int i = 0; i < N; i++) {
        Y[i]= (i + 0.5) * (Ymax - Ymin) / N;
    }
    double norm = sqrt(innerProduct(0, 0));
    for(int i = 0; i < N * N; i++) {
        psi[0][2*i] = psi[0][2*i]/norm/2;
    }
}

void SingleProcess::applyHamiltonian() {
    
    // Grid Spacing
    double a = (Ymax - Ymin) / N;
    double a2 = a * a;
    
    // First term in Hamiltonian: Discrete Laplacian (not labeled)
    // If psi[counts+1][i] is at boundary, boundary vectors are incorporated
    // Second term in Hamiltonian: First order derivative (labeled)
    // Real parts and imaginary parts are coupled;
    // Third term in Hamiltonian: Only diagonal (labeled)
    
    for (int i = 0; i < N * N; i++) {
        psi[counts+1][2*i] = 0.0;
        psi[counts+1][2*i+1] = 0.0;
        
        psi[counts+1][2*i] += (- h * h / (2 * m)) * (-4.0) * psi[counts][2*i] / a2;
        psi[counts+1][2*i+1] += (- h * h / (2 * m)) * (-4.0) * psi[counts][2*i+1] / a2;
        // third term:
        psi[counts+1][2*i] += (q * q / (2 * m)) * (B * B) * (Y[i/N] * Y[i/N]) * psi[counts][2*i];
        psi[counts+1][2*i+1] += (q * q / (2 * m)) * (B * B) * (Y[i/N] * Y[i/N]) * psi[counts][2*i+1];
        
        if (i - N >= 0) {
            psi[counts+1][2*i] += (- h * h / (2 * m)) * psi[counts][2*i-2*N] / a2;
            psi[counts+1][2*i+1] += (- h * h / (2 * m)) * psi[counts][2*i+1-2*N] / a2;
        }
        else {
            psi[counts+1][2*i] += (- h * h / (2 * m)) * topBoundary[2*(i % N)] / a2;
            psi[counts+1][2*i+1] += (- h * h / (2 * m)) * topBoundary[2*(i % N)+1] / a2;
        }
        if (i % N != 0) {
            psi[counts+1][2*i] += (- h * h / (2 * m)) * psi[counts][2*i-2] / a2;
            psi[counts+1][2*i+1] += (- h * h / (2 * m)) * psi[counts][2*i+1-2] / a2;
            // second term off boundary:
            psi[counts+1][2*i] += -(- q * h / m) * (-B) * Y[i/N] * psi[counts][2*i-2+1] / a / 2;
            psi[counts+1][2*i+1] += -(q * h / m) * (-B) * Y[i/N] * psi[counts][2*i-2] / a / 2;
        }
        else {
            psi[counts+1][2*i] += (- h * h / (2 * m)) * leftBoundary[2*(i / N)] / a2;
            psi[counts+1][2*i+1] += (- h * h / (2 * m)) * leftBoundary[2*(i / N)+1] / a2;
            // second term at boundary:
            psi[counts+1][2*i] += (- q * h / m) * (-B) * Y[i/N] * leftBoundary[2*(i / N)+1] / a / 2;
            psi[counts+1][2*i+1] += (q * h / m) * (-B) * Y[i/N] * leftBoundary[2*(i / N)] / a / 2;
        }
        if (i % N != N-1) {
            psi[counts+1][2*i] += (- h * h / (2 * m)) * psi[counts][2*i+2] / a2;
            psi[counts+1][2*i+1] += (- h * h / (2 * m)) * psi[counts][2*i+1+2] / a2;
            // second term off boundar:
            psi[counts+1][2*i] += (- q * h / m) * (-B) * Y[i/N] * psi[counts][2*i+2+1] / a / 2;
            psi[counts+1][2*i+1] += (q * h / m) * (-B) * Y[i/N] * psi[counts][2*i+2] / a / 2;
        }
        else {
            psi[counts+1][2*i] += (- h * h / (2 * m)) * rightBoundary[2*(i / N)] / a2;
            psi[counts+1][2*i+1] += (- h * h / (2 * m)) * rightBoundary[2*(i / N)+1] / a2;
            // second term at boundary:
            psi[counts+1][2*i] += (- q * h / m) * (-B) * Y[i/N] * rightBoundary[2*(i / N)+1] / a / 2;
            psi[counts+1][2*i+1] += (q * h / m) * (-B) * Y[i/N] * rightBoundary[2*(i / N)] / a / 2;
        }
        if (i + N < N * N) {
            psi[counts+1][2*i] += (- h * h / (2 * m)) * psi[counts][2*i+2*N] / a2;
            psi[counts+1][2*i+1] += (- h * h / (2 * m)) * psi[counts][2*i+1+2*N] / a2;
        }
        else {
            psi[counts+1][2*i] += (- h * h / (2 * m)) * bottomBoundary[2*(i % N)] / a2;
            psi[counts+1][2*i+1] += (- h * h / (2 * m)) * bottomBoundary[2*(i % N)+1] / a2;
        }
    }
    alpha[counts] = innerProduct(counts+1, counts);
//    for (int i = 0; i < N * N * 2; i++) {
//        cout <<  "psi[counts+1][" << i << "]" << psi[counts+1][i] << " " << psi[counts][i] << endl;
//    }
//    cout << "----- * ------ * -------" << alpha[counts] << endl;
}

void SingleProcess::updatePsiA() {
    
    for (int j = 0; j < 2 * N * N; j++) {
        psi[counts+1][j] -= alpha[counts] * psi[counts][j];
    }
    if (counts > 0) {
        for (int j = 0; j < 2 * N * N; j++) {
            psi[counts+1][j] -= beta[counts-1] * psi[counts-1][j];
        }
    }
    beta[counts] = innerProduct(counts+1, counts+1);
    /**********************/
}

void SingleProcess::updatePsiB() {
    for (int j = 0; j < 2 * N * N; j++) {
        psi[counts+1][j] = psi[counts+1][j] / beta[counts];
    }
    counts++;
}

double SingleProcess::innerProduct(int i, int j) {
    double result = 0.0;
    for (int k = 0; k < N * N; k++) {
        result += psi[i][2*k] * psi[j][2*k] + psi[i][2*k+1] * psi[j][2*k+1];
    }
    return result;
}

std::vector<double> SingleProcess::getTopBoundary() {
    vector<double> boundaryData(2*N);
    for (int i = 0 ; i < N; i++) {
        boundaryData[2*i] = psi[counts][2*i];
        boundaryData[2*i+1] = psi[counts][2*i+1];
    }
    return boundaryData;
}

std::vector<double> SingleProcess::getBottomBoundary() {
    vector<double> boundaryData(2*N);
    for (int i = 0 ; i < N; i++) {
        boundaryData[2*i] = psi[counts][2*N*(N-1)+2*i];
        boundaryData[2*i+1] = psi[counts][2*N*(N-1)+2*i+1];
      }
    return boundaryData;
}

std::vector<double> SingleProcess::getLeftBoundary() {
    vector<double> boundaryData(2*N);
    for (int i = 0 ; i < N; i++) {
        boundaryData[2*i] = psi[counts][2*N*i];
        boundaryData[2*i+1] = psi[counts][2*N*i+1];
     }
    return boundaryData;
}

std::vector<double> SingleProcess::getRightBoundary() {
    vector<double> boundaryData(2*N);
    for (int i = 0 ; i < N; i++) {
        boundaryData[2*i] = psi[counts][2*N*(i+1)-2];
        boundaryData[2*i+1] = psi[counts][2*N*(i+1)-1];
    }
    return boundaryData;
}

void SingleProcess::setPsi(const vector<double>& psi) {
    for (int i = 0; i < N * N; i++) {
        this->psi[counts][i] = psi[i];
    }
    counts++;
}

const double m = 1.0;
const double h = 1.0;
const double q = 1.0;






