package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// 行列を表示
func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func main() {
	a := mat.NewDense(3, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9})
	b := mat.NewDense(3, 1, []float64{1, 2, 3})
	c := mat.NewDense(3, 1, nil)
	c.Product(a, b)
	matPrint(a)
	matPrint(b)
	matPrint(c)
}
