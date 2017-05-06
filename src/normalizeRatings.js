var mathjs = require("mathjs");

module.exports = function (Y, R) {

    var [ m, n ] = Y._size;

    var Ymean = mathjs.zeros(m, 1);
    var Ynorm = mathjs.zeros.apply(null, Y._size);

    for(var i=0; i<m; i++) {

        var Ycolumn = mathjs.flatten(
            mathjs.subset(
                Y,
                mathjs.index(i, mathjs.range(0, n))
            )
        );

        var Rcolumn = mathjs.flatten(
            mathjs.subset(
                R,
                mathjs.index(i, mathjs.range(0, n))
            )
        );

        var mean = mathjs.mean(
            mathjs.filter(
                Ycolumn,
                function (value, index, matrix) {
                    return mathjs.subset(
                        Rcolumn,
                        mathjs.index.apply(null, index)
                    );
                }
            )
        );

        Ymean = mathjs.subset(Ymean, mathjs.index(i, 0), mean);

        Ynorm = mathjs.subset(
            Ynorm,
            mathjs.index(i, mathjs.range(0, Ynorm._size[1])),
            mathjs.subtract(
                Ycolumn,
                mathjs.dotMultiply(
                    mathjs.multiply(
                        mathjs.ones(Ycolumn._size),
                        mean
                    ),
                    Rcolumn
                )
            )
        );
    }

    return [ Ynorm, Ymean ];
};
