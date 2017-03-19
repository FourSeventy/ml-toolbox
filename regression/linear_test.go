package regression

import (
	"fmt"
	"testing"
)

var hypTests = []struct {
	theta0 float64
	theta1 float64
	x      float64
	y      float64
}{
	//both thetas zero
	{0, 0, 0, 0},
	{0, 0, 4, 0},
	{0, 0, 70000, 0},
	{0, 0, 1.5, 0},

	//theta 1 zero
	{1, 0, 4, 1},
	{50, 0, 33, 50},
	{700.55, 0, 1122, 700.55},

	//regular
	{5, 10, 2, 25},
	{1, 2, 50, 101},
}

func TestRunHypothesis(t *testing.T) {
	for _, testCase := range hypTests {
		linearModel := LinearModel{Theta0: testCase.theta0, Theta1: testCase.theta1}

		y := linearModel.RunHypothesis(testCase.x)

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
	trainingData := [][]float64{
		{1, 1},
		{2, 2},
		{3, 3},
		{4, 4},
		{5, 5},
		{6, 6},
		{7, 7},
		{8, 8},
		{9, 9},
		{10, 10},
		{11, 11},
		{12, 12},
		{13, 13},
		{14, 14},
		{15, 15},
		{16, 16},
		{17, 17},
		{60, 60},
		{99, 99},
		{20, 20},
	}

	model := LinearModel{0, 0}
	model.Train(trainingData, .001)
	fmt.Println(model.Theta0, model.Theta1)
	fmt.Println(model.RunHypothesis(80))

}
