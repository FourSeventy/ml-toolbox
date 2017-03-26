package dataset

import (
	"testing"
)

func TestLoadMissingFile(t *testing.T) {
	_, err := Load("../testdata/missing.csv")
	if err == nil {
		t.Error(
			"Expected: ", "Load to return an error",
			"Got: ", "No error",
		)
	}
}

func TestLoad(t *testing.T) {
	dataset, err := Load("../testdata/univariate_linear.csv")
	if err != nil {
		t.Error(err.Error())
	}

	if dataset == nil {
		t.Error(
			"Expected:", 400,
			"Got:", "nil",
		)
	} else if len(*dataset) != 400 {
		t.Error(
			"Expected:", 400,
			"Got:", len(*dataset),
		)
	}
}
