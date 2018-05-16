package main

import (
	"flag"
	"fmt"
	"image"
	//"image/color"
	"github.com/colin1806/GoMNIST"
	"image/jpeg"
	"image/png"
	"log"
	"os"
)

func write_jpeg(img GoMNIST.RawImage, width int, height int, path string) {
	im := image.NewGray(image.Rectangle{Max: image.Point{X: width, Y: height}})
	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			im.Set(x, y, img.At(x, y))
		}
	}

	file, err := os.Create(path)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	jpeg.Encode(file, im, nil)
}

func write_png(img GoMNIST.RawImage, width int, height int, path string) {
	im := image.NewGray(image.Rectangle{Max: image.Point{X: width, Y: height}})
	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			im.SetGray(x, y, img.AtGray(x, y))
		}
	}

	file, err := os.Create(path)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	png.Encode(file, im)
}

func to_image(frist_N int) {
	train_set, _, err := GoMNIST.Load("data")
	if err != nil {
		panic(err)
	}
	sweep := train_set.Sweep()
	cnt := 0
	for {
		cnt++
		img, label, status := sweep.Next()
		if status == false {
			break
		}

		jpeg_file := fmt.Sprintf("images/%d_%06d.jpeg", label, cnt)
		png_file := fmt.Sprintf("images/%d_%06d.png", label, cnt)
		write_jpeg(img, train_set.NRow, train_set.NCol, jpeg_file)
		write_png(img, train_set.NRow, train_set.NCol, png_file)

		if frist_N == cnt {
			break
		}
	}
}

func main() {

	first_n := flag.Int("N", 10, "convert first N images")
	flag.Parse()

	err := os.Mkdir("images", os.ModePerm)
	if err != nil {
		if os.IsNotExist(err) {
			fmt.Println(err)
		}
	}

	to_image(*first_n)
}
