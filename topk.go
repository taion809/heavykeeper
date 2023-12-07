package main

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"text/tabwriter"

	"github.com/spaolacci/murmur3"
)

const (
	TOPK_DECAY = 256
	GA         = 1919
)

type TopK struct {
	width       uint32
	depth       uint32
	hf          func([]byte, uint32) uint32
	bucket      [][]Bucket
	lookupTable [TOPK_DECAY]float64
	minHeap     *MinHeap
}

type HeapBucket struct {
	FP   uint32
	C    uint32
	item string
}

type MinHeap []HeapBucket

type Bucket struct {
	FP uint32
	C  uint32
}

func (h MinHeap) Len() int           { return len(h) }
func (h MinHeap) Less(i, j int) bool { return h[i].C < h[j].C }
func (h MinHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *MinHeap) Push(x any) {
	*h = append(*h, x.(HeapBucket))
}

func (h *MinHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

func NewTopK(w, d uint32) *TopK {
	t := TopK{
		width: w,
		depth: d,
		hf:    murmur3.Sum32WithSeed,
	}

	t.bucket = make([][]Bucket, d)
	for i := range t.bucket {
		t.bucket[i] = make([]Bucket, w)
	}

	for i := 0; i < TOPK_DECAY; i++ {
		t.lookupTable[i] = math.Pow(0.9, float64(i))
	}

	mh := make(MinHeap, 10)
	t.minHeap = &mh
	heap.Init(t.minHeap)

	return &t
}

func (t *TopK) itemExists(item string) int {
	FP := t.hf([]byte(item), GA)

	for i := 10 - 1; i >= 0; i-- {
		if FP == (*t.minHeap)[i].FP && (*t.minHeap)[i].item == item {
			return i
		}
	}

	return -1
}

func (t *TopK) Add(item string, increment uint32) error {
	fp := t.hf([]byte(item), GA)
	maxCount := uint32(0)

	// fmt.Println("FP: ", fp)

	for i := uint32(0); i < t.depth; i++ {
		h := t.hf([]byte(item), i)
		slot := h % t.width

		C := &t.bucket[i][slot].C
		Fi := &t.bucket[i][slot].FP

		if *C == 0 {
			*Fi = fp
			*C = increment
			maxCount = max(maxCount, increment)
		} else if *Fi == fp {
			*C += increment
			maxCount = max(maxCount, *C)
		} else {
			for localInc := increment; localInc > 0; localInc-- {
				var decay float64
				if *C < TOPK_DECAY {
					decay = t.lookupTable[*C]
				} else {
					// fmt.Println(t.lookupTable[TOPK_DECAY-1])
					// fmt.Println(*C)
					// fmt.Println(*C / (TOPK_DECAY - 1))
					// fmt.Println(*C % (TOPK_DECAY - 1))
					// fmt.Println(t.lookupTable[*C%(TOPK_DECAY-1)])
					// fmt.Println(float64(*C/(TOPK_DECAY-1)) * t.lookupTable[*C%(TOPK_DECAY-1)])

					decay = math.Pow(t.lookupTable[TOPK_DECAY-1], float64(*C/(TOPK_DECAY-1))) * t.lookupTable[*C%(TOPK_DECAY-1)]

					// fmt.Println(decay)
					// fmt.Println("------------")
				}

				if rand.Float64() < decay {
					*C--
					if *C == 0 {
						*Fi = fp
						*C = localInc
						maxCount = max(maxCount, *C)
						break
					}
				}
			}
		}
	}

	if maxCount >= uint32(t.minHeap.Len()) {
		var existing int = t.itemExists(item)
		if existing >= 0 {
			(*t.minHeap)[existing].C = maxCount
			heap.Fix(t.minHeap, existing)
		} else {
			(*t.minHeap)[0].C = maxCount
			(*t.minHeap)[0].FP = fp
			(*t.minHeap)[0].item = item
			heap.Fix(t.minHeap, 0)
		}
	}

	return nil
}

func (t *TopK) String() string {
	sb := &strings.Builder{}

	w := tabwriter.NewWriter(sb, 0, 0, 1, ' ', tabwriter.AlignRight|tabwriter.Debug)
	fmt.Fprintln(w, "Data")
	for i := uint32(0); i < t.depth; i++ {
		fmt.Fprintf(w, "%d:\t", i)
		for j := uint32(0); j < t.width; j++ {
			fmt.Fprintf(w, "[%d]\t", t.bucket[i][j])
		}

		fmt.Fprintln(w, "")
	}

	fmt.Fprintln(w, "Heap")
	for t.minHeap.Len() > 0 {
		bucket := heap.Pop(t.minHeap)
		b := bucket.(HeapBucket)
		fmt.Fprintf(w, "FP: %d\tC: %d\t Item: %s\n", b.FP, b.C, b.item)
	}

	w.Flush()

	return sb.String()
}
