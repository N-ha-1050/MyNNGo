package main

import (
	"math"

	"github.com/N-ha-1050/MyNNGo/utils"
	"gonum.org/v1/gonum/mat"
)

func middleLayer(x mat.Matrix, w mat.Matrix, b mat.Matrix) mat.Matrix {
	xr, _ := x.Dims()                // (1, n)
	_, wc := w.Dims()                // (n, m)
	xw := mat.NewDense(xr, wc, nil)  // (1, m)
	xw.Product(x, w)                 // (1, n) @ (n, m) = (1, m)
	u := mat.NewDense(xr, wc, nil)   // (1, m)
	u.Add(xw, b)                     // (1, m) + (1, m) = (1, m)
	res := mat.NewDense(xr, wc, nil) // (1, m)
	res.Apply(
		func(i, j int, v float64) float64 { // シグモイド関数
			return 1 / (1 + math.Exp(-v))
		},
		u,
	) // sigmoid((1, m)) = (1, m)
	return res
}

func outputLayer(x mat.Matrix, w mat.Matrix, b mat.Matrix) mat.Matrix {
	xr, _ := x.Dims()               // (1, n)
	_, wc := w.Dims()               // (n, m)
	xw := mat.NewDense(xr, wc, nil) // (1, m)
	xw.Product(x, w)                // (1, n) @ (n, m) = (1, m)
	u := mat.NewDense(xr, wc, nil)  // (1, m)
	u.Add(xw, b)                    // (1, m) + (1, m) = (1, m)
	return u                        // 恒等関数
}

func main() {

	// x, y座標
	xData := make([]float64, 10)
	xValue := -1.0
	for i := 0; i < 10; i++ {
		xData[i] = xValue
		xValue += 0.2
	}

	yData := make([]float64, 10)
	yValue := -1.0
	for i := 0; i < 10; i++ {
		yData[i] = yValue
		yValue += 0.2
	}

	x := mat.NewVecDense(10, xData)
	y := mat.NewVecDense(10, yData)

	z := mat.NewDense(10, 10, nil) // 出力を格納する10x10のグリッド

	// 重み
	wIm := mat.NewDense(2, 2, []float64{4.0, 4.0, 4.0, 4.0}) // 中間層2x2の行列
	wMo := mat.NewDense(2, 1, []float64{1.0, -1.0})          // 出力層2x1の行列

	// バイアス
	bIm := mat.NewDense(1, 2, []float64{3.0, -3.0}) // 中間層
	bMo := mat.NewDense(1, 1, []float64{0.1})       // 出力層

	// グリッドの各マスでニューロンの演算
	for i := 0; i < 10; i++ {
		for j := 0; j < 10; j++ {

			// 順伝播
			inp := mat.NewDense(1, 2, []float64{x.At(i, 0), y.At(j, 0)}) // 入力層
			mid := middleLayer(inp, wIm, bIm)                            // 中間層
			out := outputLayer(mid, wMo, bMo)                            // 出力層

			// グリッドにNNの出力を格納
			z.Set(j, i, out.At(0, 0))
		}
	}

	utils.MatPrint(z)

	// グリッドの表示
	utils.MatSaveGrayImage(z, "sample.jpg")
}
