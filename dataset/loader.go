package dataset

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
)

//Load loads in a dataset from a csv file located at filepath
func Load(filepath string) ([][]float64, error) {
	//open the file
	file, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("error opening file %v: %v", filepath, err)
	}
	defer file.Close()

	//parse the csv data set into slice of strings
	reader := csv.NewReader(file)
	fileContents, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("error parsing file %v: %v", filepath, err)
	}

	//parse strings into floats
	dataset := [][]float64{}
	for _, fileRow := range fileContents {
		datarow := []float64{}
		//parse the row into a row of floats
		for _, elem := range fileRow {
			elemFloat, err := strconv.ParseFloat(elem, 64)
			if err != nil {
				return nil, fmt.Errorf("error parsing file %v: %v", filepath, err)
			}
			datarow = append(datarow, elemFloat)
		}
		//append parsed row to dataset
		dataset = append(dataset, datarow)
	}

	//return array
	return dataset, nil
}
