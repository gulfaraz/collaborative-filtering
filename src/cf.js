var fs = require("fs");
var path = require("path");

var mathjs = require("mathjs");

var collaborativeFilteringCostFunction
    = require("./collaborativeFilteringCostFunction");
var checkCostFunction = require("./checkCostFunction");
var normalizeRatings = require("./normalizeRatings");
var fmincg = require("ml-util/fmincg");

var randnBoxMuller = require("ml-util/randnBoxMuller");

var Y = require("../data/Y");
Y = mathjs.matrix(Y);
var R = require("../data/R");
R = mathjs.matrix(R);

var X = require("../data/X");
X = mathjs.matrix(X);
var Theta = require("../data/Theta");
Theta = mathjs.matrix(Theta);

module.exports = function () {
    console.log("Starting Collaborative Filtering...");

    var movieOneRatings = mathjs.reshape(
        mathjs.subset(
            Y,
            mathjs.index(
                0,
                mathjs.range(
                    0,
                    Y._size[1]
                )
            )
        ),
        [ Y._size[1] ]
    );

    var movieOneUserRatings = mathjs.filter(
        movieOneRatings,
        function (value, index, matrix) {
            return mathjs.subset(R, mathjs.index(0, index));
        }
    );

    var movieOneAverageRating = mathjs.mean(movieOneUserRatings);

    console.log("Average rating for movie 1 (Toy Story): "
        + movieOneAverageRating.toFixed(1) + " / 5");

    var numberOfUsers = 4;
    var numberOfFeatures = 3;
    var numberOfMovies = 5;

    var YSample = mathjs.resize(
        Y,
        [ numberOfMovies, numberOfUsers ]
    );

    var RSample = mathjs.resize(
        R,
        [ numberOfMovies, numberOfUsers ]
    );

    var XSample = mathjs.resize(
        X,
        [ numberOfMovies, numberOfFeatures ]
    );

    var ThetaSample = mathjs.resize(
        Theta,
        [ numberOfUsers, numberOfFeatures ]
    );

    var parameters = mathjs.concat(
        mathjs.reshape(
            XSample,
            [ XSample._size[0] * XSample._size[1], 1 ]
        ),
        mathjs.reshape(
            ThetaSample,
            [ ThetaSample._size[0] * ThetaSample._size[1], 1 ]
        ),
        0
    );

    var [ J, grad ] = collaborativeFilteringCostFunction(
        mathjs.clone(parameters),
        YSample,
        RSample,
        numberOfUsers,
        numberOfMovies,
        numberOfFeatures,
        0
    );

    console.log("Cost: ", J, " <- value should be about 22.22\n");

    checkCostFunction();

    [ J, grad ] = collaborativeFilteringCostFunction(
        mathjs.clone(parameters),
        YSample,
        RSample,
        numberOfUsers,
        numberOfMovies,
        numberOfFeatures,
        1.5
    );

    console.log("Cost: ", J, " <- value should be about 31.34\n");

    checkCostFunction(1.5);

    var movieList = fs.readFileSync(
        path.join(__dirname, "..", "data", "movieIds.txt")
    ).toString();
    movieList = movieList.trim().split("\n").map(function (item) {
        item = item.split(" ");
        item.shift();
        return item.join(" ");
    });

    var myRatings = mathjs.zeros(Y._size[0], 1);

    myRatings = mathjs.subset(myRatings, mathjs.index(0, 0), 4);
    myRatings = mathjs.subset(myRatings, mathjs.index(97, 0), 2);
    myRatings = mathjs.subset(myRatings, mathjs.index(6, 0), 3);
    myRatings = mathjs.subset(myRatings, mathjs.index(11, 0), 5);
    myRatings = mathjs.subset(myRatings, mathjs.index(53, 0), 4);
    myRatings = mathjs.subset(myRatings, mathjs.index(63, 0), 5);
    myRatings = mathjs.subset(myRatings, mathjs.index(65, 0), 3);
    myRatings = mathjs.subset(myRatings, mathjs.index(68, 0), 5);
    myRatings = mathjs.subset(myRatings, mathjs.index(182, 0), 4);
    myRatings = mathjs.subset(myRatings, mathjs.index(225, 0), 5);
    myRatings = mathjs.subset(myRatings, mathjs.index(354, 0), 5);

    mathjs.forEach(myRatings, function(value, index, matrix) {
        if(value > 0) {
            console.log(
                "Provided rating "
                + value.toFixed(1)
                + " for movie "
                + movieList[index[0]]
            );
        }
    });

    Y = mathjs.concat(myRatings, Y);

    var myRatingsProvided = mathjs.isPositive(myRatings);

    mathjs.map(
        myRatingsProvided,
        function convertBooleanToInteger(value, index, matrix) {
            myRatingsProvided = mathjs.subset(
                myRatingsProvided,
                mathjs.index.apply(null, index),
                +value
            );
        }
    );

    R = mathjs.concat(myRatingsProvided , R);

    var [ Ynorm, Ymean ] = normalizeRatings(Y, R);

    numberOfUsers = Y._size[1];
    numberOfMovies = Y._size[0];
    numberOfFeatures = 10;

    var XInitial = mathjs.matrix(
        mathjs.random([ numberOfMovies, numberOfFeatures ])
    );

    mathjs.map(XInitial, function (value, index, matrix) {
        XInitial = mathjs.subset(
            XInitial,
            mathjs.index.apply(null, index),
            randnBoxMuller()
        );
    });

    var ThetaInitial = mathjs.matrix(
        mathjs.random([ numberOfUsers, numberOfFeatures ])
    );

    mathjs.map(ThetaInitial, function (value, index, matrix) {
        ThetaInitial = mathjs.subset(
            ThetaInitial,
            mathjs.index.apply(null, index),
            randnBoxMuller()
        );
    });

    parameters = mathjs.concat(
        mathjs.reshape(
            XInitial,
            [ XInitial._size[0] * XInitial._size[1], 1 ]
        ),
        mathjs.reshape(
            ThetaInitial,
            [ ThetaInitial._size[0] * ThetaInitial._size[1], 1 ]
        ),
        0
    );

    var fmincgOptions = { "maxIterations" : 100 };

    var lambda = 10;

    var theta = fmincg(function (t) {
        return collaborativeFilteringCostFunction(
            t,
            Ynorm,
            R,
            numberOfUsers,
            numberOfMovies,
            numberOfFeatures,
            lambda
        );
    }, parameters, fmincgOptions);

    var XOptimized = mathjs.subset(
        theta[0],
        mathjs.index(
            mathjs.range(0, numberOfMovies * numberOfFeatures),
            0
        )
    );

    XOptimized = mathjs.reshape(
        XOptimized,
        [ numberOfMovies, numberOfFeatures ]
    );

    var ThetaOptimized = mathjs.subset(
        theta[0],
        mathjs.index(
            mathjs.range(
                numberOfMovies * numberOfFeatures,
                (numberOfMovies * numberOfFeatures)
                + (numberOfUsers * numberOfFeatures)
            ),
            0
        )
    );

    ThetaOptimized = mathjs.reshape(
        ThetaOptimized,
        [ numberOfUsers, numberOfFeatures ]
    );

    var predictedValues = mathjs.multiply(
        XOptimized,
        mathjs.transpose(ThetaOptimized)
    );

    var myPredictions = mathjs.add(
        mathjs.subset(
            predictedValues,
            mathjs.index(
                mathjs.range(0, predictedValues._size[0]),
                0
            )
        ),
        Ymean
    );

    myPredictions = mathjs.flatten(myPredictions);

    var myPredictionsWithIndexing = [];

    mathjs.map(myPredictions, function (value, index, matrix) {
        myPredictionsWithIndexing.push({
            index: index,
            rating: mathjs.round(value, 1)
        });
    });

    myPredictionsWithIndexing = mathjs.sort(
        myPredictionsWithIndexing,
        function (a, b) {
            return b.rating - a.rating;
        }
    );

    mathjs.forEach(
        myPredictionsWithIndexing,
        function(value, index, matrix) {
            console.log(
                "Predicting rating "
                + value.rating.toFixed(1)
                + " for movie "
                + movieList[value.index]
            );
        }
    );

    console.log("Collaborative Filtering Complete");
};
