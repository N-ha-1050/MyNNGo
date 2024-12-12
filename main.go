package main

import (
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"math"
	"os"

	"gonum.org/v1/gonum/mat"
)

// 行列を表示
func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

// 各要素の値が0-1の行列をグレースケールでJpeg画像に保存
func matSaveGrayImage(X mat.Matrix, name string) {
	r, c := X.Dims()
	img := image.NewGray(image.Rect(0, 0, c, r))
	for i := 0; i < c; i++ {
		for j := 0; j < r; j++ {
			img.SetGray(i, j, color.Gray{Y: uint8(X.At(j, i) * 256)})
		}
	}
	file, _ := os.Create(name)
	jpeg.Encode(file, img, &jpeg.Options{Quality: 100})
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

	w_x := 2.5 // x座標の入力の重み
	w_y := 3.0 // y座標の入力の重み

	bias := 0.1 // バイアス

	// グリッドの各マスでニューロンの演算
	for i := 0; i < 10; i++ {
		for j := 0; j < 10; j++ {

			// 入力と重みの積の総和 + バイアス
			u := x.At(i, 0)*w_x + y.At(j, 0)*w_y + bias

			// グリッドに出力を格納
			zValue := 1 / (1 + math.Exp(-u))
			z.Set(j, i, zValue)
		}
	}

	matPrint(z)

	// グリッドの表示
	matSaveGrayImage(z, "sample.jpg")
}
