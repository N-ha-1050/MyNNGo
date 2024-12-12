package utils

import (
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"os"

	"gonum.org/v1/gonum/mat"
)

// 行列を表示
func MatPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

// 行列をグレースケールでJpeg画像に保存
func MatSaveGrayImage(X mat.Matrix, name string) {
	maxX := mat.Max(X)
	scaler := 255 / maxX
	r, c := X.Dims()
	img := image.NewGray(image.Rect(0, 0, c, r))
	for i := 0; i < c; i++ {
		for j := 0; j < r; j++ {
			img.SetGray(i, j, color.Gray{Y: uint8(X.At(j, i) * scaler)})
		}
	}
	file, _ := os.Create(name)
	jpeg.Encode(file, img, &jpeg.Options{Quality: 100})
}
