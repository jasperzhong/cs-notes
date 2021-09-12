package main

import (
	"golang.org/x/tour/wc"
	"strings"
)

func WordCount(s string) map[string]int {
	m := make(map[string]int)
	for _, word := range strings.Split(s, " ") {
		m[word] += 1
	}
	return m
}

func main() {
	wc.Test(WordCount)
}

