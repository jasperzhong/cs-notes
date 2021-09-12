package main

import (
	"io"
	"os"
	"strings"
)

type rot13Reader struct {
	r io.Reader
}

func (rd rot13Reader) Read(b []byte) (n int, e error) {
	n, e = rd.r.Read(b)

	for i := 0; i < n; i++ {
		c := b[i]
		if (c >= 'a' && c <= 'm') || (c >= 'A' && c <= 'M') {
			b[i] += 13
		} else if (c >= 'n' && c <= 'z') || (c >= 'N' && c <= 'Z') {
			b[i] -= 13
		}
	}

	return
}

func main() {
	s := strings.NewReader("Lbh penpxrq gur pbqr!")
	r := rot13Reader{s}
	io.Copy(os.Stdout, &r)
}

