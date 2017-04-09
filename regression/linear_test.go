package regression

import (
	"github.com/fourseventy/ml-toolbox/dataset"
	"testing"
)

func TestDimentions(t *testing.T) {
	lm := LinearModel{trained: true, parameters: []float64{1, 2, 3}}
	dim, _ := lm.Dimensions()
	if dim != 2 {
		t.Error(
			"Expected: ", 2,
			"Got: ", dim,
		)
	}
}

var hypTests = []struct {
	parameters []float64
	x          []float64
	y          float64
}{
	{[]float64{0, 0}, []float64{0}, 0},
	{[]float64{0, 0}, []float64{4}, 0},
	{[]float64{1, 0}, []float64{4}, 1},
	{[]float64{50, 0}, []float64{33}, 50},
	{[]float64{5, 10}, []float64{2}, 25},
	{[]float64{1, 2}, []float64{50}, 101},
	{[]float64{1, 1, 2, 3}, []float64{2, 3, 4}, 21},
	{[]float64{1, 1, 2, 3, .5}, []float64{2, 3, 4, 10}, 26},
}

func TestRunHypothesisErrors(t *testing.T) {
	//expect dimentionality mismatch error
	lm := LinearModel{trained: true, parameters: []float64{1, 2}}
	_, err := lm.RunHypothesis([]float64{2, 3, 4})
	if err == nil {
		t.Error("Expected dimentionality mismatch error, got none")
	}

	//expect untrained error
	lm = LinearModel{trained: false, parameters: []float64{1, 2}}
	_, err = lm.RunHypothesis([]float64{2, 3, 4})
	if err == nil {
		t.Error("Expected untrained model error, got none")
	}
}

func TestRunHypothesis(t *testing.T) {
	for _, testCase := range hypTests {
		linearModel := LinearModel{trained: true, parameters: testCase.parameters}

		y, err := linearModel.RunHypothesis(testCase.x)
		if err != nil {
			t.Error(err)
		}

		if y != testCase.y {
			t.Error(
				"For:", testCase.x,
				"Expected:", testCase.y,
				"Got:", y,
			)
		}
	}
}

func TestTrain(t *testing.T) {
	//load training data
	trainingData, err := dataset.Load("../testdata/univariate_linear.csv")
	if err != nil {
		t.Error(err)
	}

	//run train
	model := NewLinearModel()
	err = model.Train(trainingData, .00001)
	if err != nil {
		t.Error(err)
	}

	//TODO: need some type of acceptance conditions
}

func TestMeanSquaredError(t *testing.T) {
	model := LinearModel{trained: true, parameters: []float64{1, 1}}

	dataSet := [][]float64{
		{6, 7},
		{1, 2},
		{0, 1},
	}

	mse, err := model.MeanSquaredError(dataSet)
	if err != nil {
		t.Error(err)
	}

	if mse != 0 {
		t.Error("MSE calculation incorrect")
	}

}
