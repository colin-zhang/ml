// There are 4 files:

// train-images-idx3-ubyte: training set images
// train-labels-idx1-ubyte: training set labels
// t10k-images-idx3-ubyte:  test set images
// t10k-labels-idx1-ubyte:  test set labels

// The training set contains 60000 examples, and the test set 10000 examples.

// The first 5000 examples of the test set are taken from the original NIST training set. The last 5000 are taken from the original NIST test set. The first 5000 are cleaner and easier than the last 5000.

// TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
// [offset] [type]          [value]          [description]
// 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
// 0004     32 bit integer  60000            number of items
// 0008     unsigned byte   ??               label
// 0009     unsigned byte   ??               label
// ........
// xxxx     unsigned byte   ??               label
// The labels values are 0 to 9.

// TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
// [offset] [type]          [value]          [description]
// 0000     32 bit integer  0x00000803(2051) magic number
// 0004     32 bit integer  60000            number of images
// 0008     32 bit integer  28               number of rows
// 0012     32 bit integer  28               number of columns
// 0016     unsigned byte   ??               pixel
// 0017     unsigned byte   ??               pixel
// ........
// xxxx     unsigned byte   ??               pixel
// Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

// TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
// [offset] [type]          [value]          [description]
// 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
// 0004     32 bit integer  10000            number of items
// 0008     unsigned byte   ??               label
// 0009     unsigned byte   ??               label
// ........
// xxxx     unsigned byte   ??               label
// The labels values are 0 to 9.

// TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
// [offset] [type]          [value]          [description]
// 0000     32 bit integer  0x00000803(2051) magic number
// 0004     32 bit integer  10000            number of images
// 0008     32 bit integer  28               number of rows
// 0012     32 bit integer  28               number of columns
// 0016     unsigned byte   ??               pixel
// 0017     unsigned byte   ??               pixel
// ........
// xxxx     unsigned byte   ??               pixel
// Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
	//"sync/atomic"
)

var (
	mnist_url = "http://yann.lecun.com/exdb/mnist"
)

var mnist_train_files = []string{
	"train-images-idx3-ubyte.gz",
	"train-labels-idx1-ubyte.gz",
	"t10k-images-idx3-ubyte.gz",
	"t10k-labels-idx1-ubyte.gz",
}

func download(report chan<- string, file string) {

	url := fmt.Sprintf("%s/%s", mnist_url, file)
	res, err := http.Get(url)
	if err != nil {
		panic(err)
	}
	f, err := os.Create("data/" + file)
	if err != nil {
		panic(err)
	}

	io.Copy(f, res.Body)
	report <- fmt.Sprintf(file)
}

func main() {

	report := make(chan string)

	err := os.Mkdir("data", os.ModePerm)
	if err != nil {
		if os.IsNotExist(err) {
			fmt.Println(err)
		}
	}

	for _, s := range mnist_train_files {
		go download(report, s)
	}

	for i := 0; i < len(mnist_train_files); i++ {
		//select需要满足case才能退出,相当是个循环
		select {
		case msg1 := <-report:
			fmt.Println(msg1 + " downloaded!")
		case <-time.After(time.Second * 200):
			fmt.Println("time out !")
		}
	}
}
