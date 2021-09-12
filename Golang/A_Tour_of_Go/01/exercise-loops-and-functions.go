package main

import (
	"fmt"
	"math"
)

func Sqrt(x float64) float64 {
	eps := 1e-6
	y, z := 1.0, 1.0
	for {
		z -= (z*z - x) / (2 * z)
		if delta := math.Abs(z - y); delta < eps {
			return z
		} else {
			y = z
		}
	}
}

func main() {
	num := 2021.9
	my_sqrt := Sqrt(num)
	std_sqrt := math.Sqrt(num)
	fmt.Printf("num = %f, my_sqrt = %f, std_sqrt = %f, diff = %f", num, my_sqrt, std_sqrt, math.Abs(my_sqrt-std_sqrt))
}

