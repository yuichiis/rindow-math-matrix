<?php
namespace RindowTest\Math\Matrix\LinearAlgebraTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use ArrayObject;
use SplFixedArray;
use InvalidArgumentException;

class Test extends TestCase
{
    protected $equalEpsilon = 1e-04;

    public function newMatrixOperator()
    {
        $mo = new MatrixOperator();
        if(extension_loaded('rindow_openblas')) {
            $mo->blas()->forceBlas(true);
            $mo->math()->forceMath(true);
        }
        return $mo;
    }

    public function equalTest($a,$b)
    {
        $mo = $this->newMatrixOperator();
        if($a instanceof NDArray) {
            if(!($b instanceof NDArray))
                throw new InvalidArgumentException('NDArrays must be of the same type.');
            if($a->shape()!=$b->shape())
                return false;
            $delta = $mo->zerosLike($b);
            $this->la->copy($b,$delta);
            $this->la->axpy($a,$delta,-1.0);
            $delta = $this->la->asum($delta);
        } elseif(is_numeric($a)) {
            if(!is_numeric($b))
                throw new InvalidArgumentException('Values must be of the same type.');
            $delta = abs($a - $b);
        } elseif(is_bool($a)) {
            if(!is_bool($b))
                throw new InvalidArgumentException('Values must be of the same type.');
            $delta = ($a==$b)? 0 : 1;
        } else {
            throw new InvalidArgumentException('Values must be DNArray or float or int.');
        }

        if($delta < $this->equalEpsilon) {
            return true;
        } else {
            return false;
        }
    }

    public function testAlloc()
    {
        $mo = $this->newMatrixOperator();
        $x = $mo->la()->alloc([2,3]);
        $this->assertEquals([2,3],$x->shape());
        $this->assertEquals(NDArray::float32,$x->dtype());
        $this->assertEquals(6,$x->size());
        $this->assertEquals(6,count($x->buffer()));
    }

    public function testScal()
    {
        $mo = $this->newMatrixOperator();
        $x = $mo->array([[1,2,3],[4,5,6],[7,8,9]],NDArray::float32);
        $mo->la()->scal(2,$x);
        $this->assertEquals([[2,4,6],[8,10,12],[14,16,18]],$x->toArray());
    }

    /**
    *    Y := alpha * X + Y
    */
    public function testaxpy()
    {
        $mo = $this->newMatrixOperator();
        $x = $mo->array([[1,2,3],[4,5,6]],NDArray::float32);
        $y = $mo->array([[10,20,30],[40,50,60]],NDArray::float32);
        $mo->la()->axpy($x,$y,2);
        $this->assertEquals([[12,24,36],[48,60,72]],$y->toArray());
    }

    /**
    *    ret := X^t Y = x_1 * y_1 + ... + x_n * y_n
    */
    public function testdot()
    {
        $mo = $this->newMatrixOperator();
        $x = $mo->array([[1,2,3],[4,5,6]],NDArray::float32);
        $y = $mo->array([[10,20,30],[40,50,60]],NDArray::float32);
        $ret = $mo->la()->dot($x,$y);
        $this->assertEquals(1*10+2*20+3*30+4*40+5*50+6*60,$ret);
    }

    /**
    *    ret := |x_1| + ... + |x_n|
    */
    public function testasum()
    {
        $mo = $this->newMatrixOperator();
        $x = $mo->array([[-1,2,-3],[-4,5,-6]],NDArray::float32);
        $ret = $mo->la()->asum($x);
        $this->assertEquals(1+2+3+4+5+6,$ret);
    }

    /**
    *    ret := arg max X(i)
    */
    public function testimax()
    {
        $mo = $this->newMatrixOperator();
        $x = $mo->array([[-1,2,-3],[-4,5,-6]],NDArray::float32);
        $ret = $mo->la()->imax($x);
        $this->assertEquals(4,$ret);
    }

    /**
    *    ret := arg max |X(i)|
    */
    public function testiamax()
    {
        $mo = $this->newMatrixOperator();
        $x = $mo->array([[-1,2,-3],[-4,5,-6]],NDArray::float32);
        $ret = $mo->la()->iamax($x);
        $this->assertEquals(5,$ret);
    }

    /**
    *    ret := arg min X(i)
    */
    public function testimin()
    {
        $mo = $this->newMatrixOperator();
        $x = $mo->array([[-1,2,-3],[-4,5,-6]],NDArray::float32);
        $ret = $mo->la()->imin($x);
        $this->assertEquals(5,$ret);
    }

    /**
    *    ret := arg min |X(i)|
    */
    public function testiamin()
    {
        $mo = $this->newMatrixOperator();
        $x = $mo->array([[-1,2,-3],[-4,5,-6]],NDArray::float32);
        $ret = $mo->la()->iamin($x);
        $this->assertEquals(0,$ret);
    }

    /**
    *    ret := max X(i)
    */
    public function testmax()
    {
        $mo = $this->newMatrixOperator();
        $x = $mo->array([[-1,2,-3],[-4,5,-6]],NDArray::float32);
        $ret = $mo->la()->max($x);
        $this->assertEquals(5,$ret);
    }

    /**
    *    ret := max |X(i)|
    */
    public function testamax()
    {
        $mo = $this->newMatrixOperator();
        $x = $mo->array([[-1,2,-3],[-4,5,-6]],NDArray::float32);
        $ret = $mo->la()->amax($x);
        $this->assertEquals(-6,$ret);
    }

    /**
    *    ret := min X(i)
    */
    public function testmin()
    {
        $mo = $this->newMatrixOperator();
        $x = $mo->array([[-1,2,-3],[-4,5,-6]],NDArray::float32);
        $ret = $mo->la()->min($x);
        $this->assertEquals(-6,$ret);
    }

    /**
    *    ret := min |X(i)|
    */
    public function testamin()
    {
        $mo = $this->newMatrixOperator();
        $x = $mo->array([[-1,2,-3],[-4,5,-6]],NDArray::float32);
        $ret = $mo->la()->amin($x);
        $this->assertEquals(-1,$ret);
    }

    /**
    *    Y := X
    */
    public function testcopy()
    {
        $mo = $this->newMatrixOperator();
        $x = $mo->array([[-1,2,-3],[-4,5,-6]],NDArray::float32);
        $y = $mo->zerosLike($x);
        $ret = $mo->la()->copy($x,$y);
        $this->assertEquals([[-1,2,-3],[-4,5,-6]],$y->toArray());
    }

    public function testGemvNormal()
    {
        $mo = $this->newMatrixOperator();
        $A = $mo->array([[1,2,3],[4,5,6]]);
        $X = $mo->array([100,10,1]);

        $Y = $mo->la()->gemv($A,$X);
        $this->assertEquals(
            [123,456]
        ,$Y->toArray());
    }

    public function testGemvTranspose()
    {
        $mo = $this->newMatrixOperator();
        $A = $mo->array([[1,2,3],[4,5,6]]);
        $X = $mo->array([10,1]);

        $Y = $mo->la()->gemv($A,$X,null,null,null,$trans=true);
        $this->assertEquals(
            [14,25,36]
        ,$Y->toArray());
    }

    public function testGemmNormal()
    {
        $mo = $this->newMatrixOperator();
        $A = $mo->array([[1,2,3],[4,5,6],[7,8,9]]);
        $B = $mo->array([[1,0,0],[0,1,0],[0,0,1]]);

        $C = $mo->la()->gemm($A,$B);
        $this->assertEquals([
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ],$C->toArray());
    }

    public function testGemmScaleAlpha()
    {
        $mo = $this->newMatrixOperator();
        $A = $mo->array([[1,2,3],[4,5,6],[7,8,9]]);
        $B = $mo->array([[1,0,0],[0,1,0],[0,0,1]]);

        $C = $mo->la()->gemm($A,$B,10);
        $this->assertEquals([
            [10,20,30],
            [40,50,60],
            [70,80,90]
        ],$C->toArray());
    }

    public function testGemmScaleBeta()
    {
        $mo = $this->newMatrixOperator();
        $A = $mo->array([[1,2,3],[4,5,6],[7,8,9]]);
        $B = $mo->array([[1,0,0],[0,1,0],[0,0,1]]);

        $C = $mo->ones([$A->shape()[0],$B->shape()[1]]);
        $mo->la()->gemm($A,$B,null,10,$C);
        $this->assertEquals([
            [11,12,13],
            [14,15,16],
            [17,18,19]
        ],$C->toArray());
    }

    public function testGemmTransposeSquareA()
    {
        $mo = $this->newMatrixOperator();
        $A = $mo->array([[1,2,3],[4,5,6],[7,8,9]]);
        $B = $mo->array([[1,0,0],[0,1,0],[0,0,1]]);

        $C = $mo->la()->gemm($A,$B,null,null,null,$transA=true);
        $this->assertEquals([
            [1,4,7],
            [2,5,8],
            [3,6,9]
        ],$C->toArray());
    }

    public function testGemmTransposeSquareB()
    {
        $mo = $this->newMatrixOperator();
        $A = $mo->array([[1,0,0],[0,1,0],[0,0,1]]);
        $B = $mo->array([[1,2,3],[4,5,6],[7,8,9]]);

        $C = $mo->la()->gemm($A,$B,null,null,null,null,$transB=true);
        $this->assertEquals([
            [1,4,7],
            [2,5,8],
            [3,6,9]
        ],$C->toArray());
    }

    public function testGemmNoTransRectangleA23()
    {
        $mo = $this->newMatrixOperator();
        $A = $mo->array([[1,2,3],[4,5,6]]);
        $B = $mo->array([[1,0,0],[0,1,0],[0,0,1]]);

        $C = $mo->la()->gemm($A,$B);
        $this->assertEquals([
            [1,2,3],
            [4,5,6],
        ],$C->toArray());
    }

    public function testGemmTransposeRectangleA32()
    {
        $mo = $this->newMatrixOperator();
        $A = $mo->array([[1,2],[3,4],[5,6]]);
        $B = $mo->array([[1,0,0],[0,1,0],[0,0,1]]);
        $C = $mo->la()->gemm($A,$B,null,null,null,$transA=true);
        $this->assertEquals([
            [1,3,5],
            [2,4,6],
        ],$C->toArray());
    }

    public function testGemmNoTransRectangleB32()
    {
        $mo = $this->newMatrixOperator();
        $A = $mo->array([[1,0,0],[0,1,0],[0,0,1]]);
        $B = $mo->array([[1,2],[3,4],[5,6]]);
        $C = $mo->la()->gemm($A,$B);
        $this->assertEquals([
            [1,2],
            [3,4],
            [5,6],
        ],$C->toArray());
    }

    public function testGemmTransposeRectangleB23()
    {
        $mo = $this->newMatrixOperator();
        $A = $mo->array([[1,0,0],[0,1,0],[0,0,1]]);
        $B = $mo->array([[1,2,3],[4,5,6]]);
        $C = $mo->la()->gemm($A,$B,null,null,null,null,$transB=true);
        $this->assertEquals([
            [1,4],
            [2,5],
            [3,6],
        ],$C->toArray());
    }

    public function testGemmUnmatchShapeNoTransRectangleA32()
    {
        $mo = $this->newMatrixOperator();
        $A = $mo->array([[1,2],[3,4],[5,6]]);
        $B = $mo->array([[1,0,0],[0,1,0],[0,0,1]]);
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('The number of columns in "A" and the number of rows in "B" must be the same');
        $C = $mo->la()->gemm($A,$B);
    }

    public function testGemmUnmatchShapeTransposeRectangleA23()
    {
        $mo = $this->newMatrixOperator();
        $A = $mo->array([[1,2,3],[4,5,6]]);
        $B = $mo->array([[1,0,0],[0,1,0],[0,0,1]]);
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('The number of columns in "A" and the number of rows in "B" must be the same');
        $C = $mo->la()->gemm($A,$B,null,null,null,$transA=true);
    }

    public function testGemmUnmatchShapeNoTransRectangleB23()
    {
        $mo = $this->newMatrixOperator();
        $A = $mo->array([[1,0,0],[0,1,0],[0,0,1]]);
        $B = $mo->array([[1,2,3],[4,5,6]]);
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('The number of columns in "A" and the number of rows in "B" must be the same');
        $C = $mo->la()->gemm($A,$B);
    }

    public function testGemmUnmatchShapeTransposeRectangleB32()
    {
        $mo = $this->newMatrixOperator();
        $A = $mo->array([[1,0,0],[0,1,0],[0,0,1]]);
        $B = $mo->array([[1,2],[3,4],[5,6]]);
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('The number of columns in "A" and the number of rows in "B" must be the same');
        $C = $mo->la()->gemm($A,$B,null,null,null,null,$transB=true);
    }

    public function testGemmUnmatchOutputShapeNoTransA()
    {
        $mo = $this->newMatrixOperator();
        $A = $mo->array([[1,2,3],[4,5,6]]);
        $B = $mo->array([[1,0,0],[0,1,0],[0,0,1]]);

        $C = $mo->zeros([3,3]);
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('"A" and "C" must have the same number of rows."B" and "C" must have the same number of columns');
        $mo->la()->gemm($A,$B,null,null,$C);
    }

    public function testGemmUnmatchOutputShapeNoTransB()
    {
        $mo = $this->newMatrixOperator();
        $A = $mo->array([[1,2,3],[4,5,6]]);
        $B = $mo->array([[1,0,0],[0,1,0],[0,0,1]]);

        $C = $mo->zeros([2,2]);
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('"A" and "C" must have the same number of rows."B" and "C" must have the same number of columns');
        $mo->la()->gemm($A,$B,null,null,$C);
    }

    public function testGemmUnmatchOutputShapeTransposeA()
    {
        $mo = $this->newMatrixOperator();
        $A = $mo->array([[1,2],[3,4],[5,6]]);
        $B = $mo->array([[1,0,0],[0,1,0],[0,0,1]]);

        $C = $mo->zeros([3,3]);
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('"A" and "C" must have the same number of rows."B" and "C" must have the same number of columns');
        $mo->la()->gemm($A,$B,null,null,$C,$transA=true);
    }

    public function testGemmUnmatchOutputShapeTransposeB()
    {
        $mo = $this->newMatrixOperator();
        $A = $mo->array([[1,0,0],[0,1,0],[0,0,1]]);
        $B = $mo->array([[1,2,3],[4,5,6]]);

        $C = $mo->zeros([3,3]);
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('"A" and "C" must have the same number of rows."B" and "C" must have the same number of columns');
        $mo->la()->gemm($A,$B,null,null,$C,null,$transB=true);
    }

    public function testIncrement()
    {
        $mo = $this->newMatrixOperator();
        $X = $mo->array([[1,2,3],[4,5,6]]);

        // X := X + 1
        $mo->la()->increment($X,1.0);
        $this->assertEquals(
            [[2,3,4],[5,6,7]]
        ,$X->toArray());

        // X := 8 - X
        $mo->la()->increment($X,8.0,-1.0);
        $this->assertEquals(
            [[6,5,4],[3,2,1]]
        ,$X->toArray());

        // X := 2 * X
        $mo->la()->increment($X,null,2.0);
        $this->assertEquals(
            [[12,10,8],[6,4,2]]
        ,$X->toArray());
    }

    public function testReciprocal()
    {
        $mo = $this->newMatrixOperator();

        // X := 1 / X
        $X = $mo->array([[1,2,4],[8,16,32]]);
        $mo->la()->reciprocal($X);
        $this->assertEquals(
            [[1,0.5,0.25],[0.125,0.0625, 0.03125]]
        ,$X->toArray());

        // X := 1 / (X + 1)
        $X = $mo->array([[0,1,3],[7,15,31]]);
        $mo->la()->reciprocal($X,1.0);
        $this->assertEquals(
            [[1,0.5,0.25],[0.125,0.0625, 0.03125]]
        ,$X->toArray());

        // X := 1 / (32 - X)
        $X = $mo->array([[31,30,28],[24,16,0]]);
        $mo->la()->reciprocal($X,32,-1.0);
        $this->assertEquals(
            [[1,0.5,0.25],[0.125,0.0625, 0.03125]]
        ,$X->toArray());
    }

    public function testMaximum()
    {
        $mo = $this->newMatrixOperator();

        // X := maximum(1,X)
        $X = $mo->array([[-1,0,1],[2,3,4]]);
        $mo->la()->maximum(1,$X);
        $this->assertEquals(
            [[1,1,1],[2,3,4]]
        ,$X->toArray());
    }

    public function testMinimum()
    {
        $mo = $this->newMatrixOperator();

        // X := minimum(1,X)
        $X = $mo->array([[-1,0,1],[2,3,4]]);
        $mo->la()->minimum(1,$X);
        $this->assertEquals(
            [[-1,0,1],[1,1,1]]
        ,$X->toArray());
    }

    public function testGreater()
    {
        $mo = $this->newMatrixOperator();

        // X := greater(1,X)
        $X = $mo->array([[-1,0,1],[2,3,4]]);
        $mo->la()->greater(1,$X);
        $this->assertEquals(
            [[0,0,0],[1,1,1]]
        ,$X->toArray());
    }

    public function testLess()
    {
        $mo = $this->newMatrixOperator();

        // X := less(1,X)
        $X = $mo->array([[-1,0,1],[2,3,4]]);
        $mo->la()->less(1,$X);
        $this->assertEquals(
            [[1,1,0],[0,0,0]]
        ,$X->toArray());
    }

    public function testMultiply()
    {
        $mo = $this->newMatrixOperator();

        // Y := X(i) * Y(i)
        $X = $mo->array([[1,2,3],[4,5,6]]);
        $Y = $mo->array([[1,10,100],[-1,-10,-100]]);
        $mo->la()->multiply($X,$Y);
        $this->assertEquals(
            [[1,2,3],[4,5,6]]
        ,$X->toArray());
        $this->assertEquals(
            [[1,20,300],[-4,-50,-600]]
        ,$Y->toArray());

        // broadcast
        $X = $mo->array([1,2,3]);
        $Y = $mo->array([[1,10,100],[-1,-10,-100]]);
        $mo->la()->multiply($X,$Y);
        $this->assertEquals(
            [1,2,3]
        ,$X->toArray());
        $this->assertEquals(
            [[1,20,300],[-1,-20,-300]]
        ,$Y->toArray());

        // transpose and broadcast
        $X = $mo->array([1,2,3]);
        $Y = $mo->array([[1,10],[100,-1],[-10,-100]]);
        $mo->la()->multiply($X,$Y,$trans=true);
        $this->assertEquals(
            [1,2,3]
        ,$X->toArray());
        $this->assertEquals(
            [[1,10],[200,-2],[-30,-300]]
        ,$Y->toArray());
    }

    public function testAdd()
    {
        $mo = $this->newMatrixOperator();

        // Y := X(i) * Y(i)
        $X = $mo->array([[1,2,3],[4,5,6]]);
        $Y = $mo->array([[1,10,100],[-1,-10,-100]]);
        $mo->la()->add($X,$Y);
        $this->assertEquals(
            [[1,2,3],[4,5,6]]
        ,$X->toArray());
        $this->assertEquals(
            [[2,12,103],[3,-5,-94]]
        ,$Y->toArray());

        // broadcast and alpha = -1
        $X = $mo->array([1,2,3]);
        $Y = $mo->array([[1,10,100],[-1,-10,-100]]);
        $mo->la()->add($X,$Y,-1);
        $this->assertEquals(
            [1,2,3]
        ,$X->toArray());
        $this->assertEquals(
            [[0,8,97],[-2,-12,-103]]
        ,$Y->toArray());

        // transpose and broadcast
        $X = $mo->array([1,2,3]);
        $Y = $mo->array([[1,10],[100,-1],[-10,-100]]);
        $mo->la()->add($X,$Y,null,$trans=true);
        $this->assertEquals(
            [1,2,3]
        ,$X->toArray());
        $this->assertEquals(
            [[2,11],[102,1],[-7,-97]]
        ,$Y->toArray());
    }

    public function testSquare()
    {
        $mo = $this->newMatrixOperator();

        // X := X ^ 2
        $X = $mo->array([[1,2,3],[4,5,6]]);
        $mo->la()->square($X);
        $this->assertEquals(
            [[1,4,9],[16,25,36]]
        ,$X->toArray());
    }

    public function testSqrt()
    {
        $mo = $this->newMatrixOperator();

        // X := sqrt(X)
        $X = $mo->array([[1,4,9],[16,25,36]]);
        $mo->la()->sqrt($X);
        $this->assertEquals(
            [[1,2,3],[4,5,6]]
        ,$X->toArray());
    }

    public function testRsqrt()
    {
        $mo = $this->newMatrixOperator();

        // X := 1 / sqrt(X)
        $X = $mo->array([[1,4],[16,64]]);
        $mo->la()->rsqrt($X);
        $this->assertEquals(
            [[1/1,1/2],[1/4,1/8]]
        ,$X->toArray());

        // X := 1 / ( 1 - sqrt(X))
        $X = $mo->array([[10,40],[80,160]]);
        $mo->la()->rsqrt($X,1,-1);

        $mo->la()->reciprocal($X);
        $mo->la()->increment($X,1,-1);
        $mo->la()->square($X);

        $this->assertEquals(
            [[10,40],[80,160]]
        ,$X->toArray());
    }

    public function testPow()
    {
        $mo = $this->newMatrixOperator();

        // X := sqrt(X)
        $X = $mo->array([[1,2,3],[4,5,6]]);
        $mo->la()->pow($X,3);
        $this->assertEquals(
            [[1,8,27],[64,125,216]]
        ,$X->toArray());
    }

    public function testExp()
    {
        $mo = $this->newMatrixOperator();

        // X := exp(X)
        $X = $mo->array([[1,2,3],[4,5,6]]);
        $X2 = $mo->copy($X);
        $mo->la()->exp($X);
        $Y = $mo->f(function ($x) { return log($x);},$X);
        $this->assertLessThan(1e-5,$mo->asum($mo->op($X2,'-',$Y)));
    }

    public function testLog()
    {
        $mo = $this->newMatrixOperator();

        // X := log(X)
        $X = $mo->array([[1,2,3],[4,5,6]]);
        $X2 = $mo->copy($X);
        $mo->la()->log($X);
        $Y = $mo->f(function ($x) { return exp($x);},$X);
        $this->assertLessThan(1e-5,$mo->asum($mo->op($X2,'-',$Y)));
    }

    public function testDuplicate()
    {
        $mo = $this->newMatrixOperator();

        // Y := X (duplicate 2 times)
        $X = $mo->array([[1,2,3],[4,5,6]]);
        $Y = $mo->la()->duplicate($X,2);
        $this->assertEquals(
            [[1,2,3],[4,5,6]]
        ,$X->toArray());
        $this->assertEquals([
            [[1,2,3],[4,5,6]],
            [[1,2,3],[4,5,6]],
        ],$Y->toArray());

        // 1 time
        $X = $mo->array([[1,2,3],[4,5,6]]);
        $Y = $mo->la()->duplicate($X,1);
        $this->assertEquals(
            [[1,2,3],[4,5,6]]
        ,$X->toArray());
        $this->assertEquals(
            [[[1,2,3],[4,5,6]]]
        ,$Y->toArray());

        // transpose
        $X = $mo->array([[1,2,3],[4,5,6]]);
        $Y = $mo->la()->duplicate($X,2,true);
        $this->assertEquals(
            [[1,2,3],[4,5,6]]
        ,$X->toArray());
        $this->assertEquals([
            [[1,1],[2,2],[3,3]],
            [[4,4],[5,5],[6,6]],
        ],$Y->toArray());
    }

    public function testRepeat()
    {
        $mo = $this->newMatrixOperator();

        // Y := X (duplicate 2 times)
        $X = $mo->array([[1,2,3],[4,5,6]]);
        $Y = $mo->la()->repeat($X,2);
        $this->assertEquals([
            [1,2,3],
            [4,5,6]
        ],$X->toArray());
        $this->assertEquals([
            [[1,2,3],[1,2,3]],
            [[4,5,6],[4,5,6]],
        ],$Y->toArray());

        // 1 time
        $X = $mo->array([[1,2,3],[4,5,6]]);
        $Y = $mo->la()->repeat($X,1);
        $this->assertEquals(
            [[1,2,3],[4,5,6]]
        ,$X->toArray());
        $this->assertEquals([2,1,3],$Y->shape());
        $this->assertEquals(
            [[[1,2,3]],[[4,5,6]]]
        ,$Y->toArray());

        $X = $mo->array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]);
        $Y = $mo->la()->repeat($X,4);
        $this->assertEquals(
            [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]
        ,$X->toArray());
        $this->assertEquals([2,4,2,3],$Y->shape());
        $this->assertEquals([
            [[[1,2,3],[4,5,6]],
             [[1,2,3],[4,5,6]],
             [[1,2,3],[4,5,6]],
             [[1,2,3],[4,5,6]]],
            [[[7,8,9],[10,11,12]],
             [[7,8,9],[10,11,12]],
             [[7,8,9],[10,11,12]],
             [[7,8,9],[10,11,12]]],
        ],$Y->toArray());
    }

    public function testZeros()
    {
        $mo = $this->newMatrixOperator();
        $x = $mo->array([1,2,3]);
        $mo->la()->zeros($x);
        $this->assertEquals([0,0,0],$x->toArray());
    }

    public function testSelect()
    {
        $mo = $this->newMatrixOperator();

        $a = $mo->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
        ],NDArray::float32);
        $x = $mo->array([0,2],NDArray::int32);
        $y = $mo->la()->select($a,$x,$axis=0);
        $this->assertEquals([[1,2,3],[7,8,9]],$y->toArray());
        $y = $mo->la()->select($a,$x,$axis=0);
        $this->assertEquals([[1,2,3],[7,8,9]],$y->toArray());

        $a = $mo->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
        ]);
        $x = $mo->array([0,1,2,0],NDArray::int32);
        $y = $mo->la()->select($a,$x,$axis=1);
        $this->assertEquals([1,5,9,10],$y->toArray());
    }

    public function testSelectAxis0()
    {
        $mo = $this->newMatrixOperator();
        $a = $mo->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
        ],NDArray::float32);
        $x = $mo->array([0,2],NDArray::int32);
        $y = $mo->la()->select($a,$x,$axis=0);
        $this->assertEquals([[1,2,3],[7,8,9]],$y->toArray());

        $a = $mo->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
        ],NDArray::float64);
        $x = $mo->array([0,2],NDArray::int64);
        $y = $mo->la()->select($a,$x,$axis=0);
        $this->assertEquals([[1,2,3],[7,8,9]],$y->toArray());

        $a = $mo->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
        ],NDArray::int64);
        $x = $mo->array([0,2],NDArray::int64);
        $y = $mo->la()->select($a,$x,$axis=0);
        $this->assertEquals([[1,2,3],[7,8,9]],$y->toArray());

        $a = $mo->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
        ],NDArray::uint8);
        $x = $mo->array([0,2],NDArray::uint8);
        $y = $mo->la()->select($a,$x,$axis=0);
        $this->assertEquals([[1,2,3],[7,8,9]],$y->toArray());

        $a = $mo->array([1,2,3,4],NDArray::float32);
        $x = $mo->array([0,2],NDArray::int32);
        $y = $mo->la()->select($a,$x,$axis=0);
        $this->assertEquals([1,3],$y->toArray());

        $a = $mo->array([1,2,3,4],NDArray::float64);
        $x = $mo->array([0,2],NDArray::int64);
        $y = $mo->la()->select($a,$x,$axis=0);
        $this->assertEquals([1,3],$y->toArray());

        $a = $mo->array([1,2,3,4],NDArray::int64);
        $x = $mo->array([0,2],NDArray::int64);
        $y = $mo->la()->select($a,$x,$axis=0);
        $this->assertEquals([1,3],$y->toArray());

        $a = $mo->array([252,253,254,255],NDArray::uint8);
        $x = $mo->array([0,2],NDArray::uint8);
        $y = $mo->la()->select($a,$x,$axis=0);
        $this->assertEquals([252,254],$y->toArray());

        $a = $mo->full([256],255,NDArray::uint8);
        $x = $mo->array([0,2],NDArray::uint8);
        $y = $mo->la()->select($a,$x,$axis=0);
        $a2 = $mo->full([256],255,NDArray::uint8);
        $x2 = $mo->array([0,2],NDArray::uint8);
        $y2 = $mo->la()->select($a2,$x2,$axis=0);

    }

    public function testSelectAxis1()
    {
        $mo = $this->newMatrixOperator();
        $a = $mo->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
        ]);
        $x = $mo->array([0,1,2,0],NDArray::int32);
        $y = $mo->la()->select($a,$x,$axis=1);
        $this->assertEquals([1,5,9,10],$y->toArray());

        $x = $mo->array([0,1,2,0],NDArray::int64);
        $y = $mo->la()->select($a,$x,$axis=1);
        $this->assertEquals([1,5,9,10],$y->toArray());

        $x = $mo->array([0,1,2,0],NDArray::float32);
        $y = $mo->la()->select($a,$x,$axis=1);
        $this->assertEquals([1,5,9,10],$y->toArray());

        $x = $mo->array([0,1,2,0],NDArray::float64);
        $y = $mo->la()->select($a,$x,$axis=1);
        $this->assertEquals([1,5,9,10],$y->toArray());
    }

    public function testOnehot()
    {
        $mo = $this->newMatrixOperator();
        $x = $mo->array([0,1,2,0]);

        $this->assertEquals([
            [1,0,0],
            [0,1,0],
            [0,0,1],
            [1,0,0],
        ],$mo->la()->onehot($x,3)->toArray());

        $y = $mo->ones([4,3]);
        $this->assertEquals([
            [-1, 1, 1],
            [ 1,-1, 1],
            [ 1, 1,-1],
            [-1, 1, 1],
        ],$mo->la()->onehot($x,3,-2,$y)->toArray());
    }

    public function testReduceSum()
    {
        $mo = $this->newMatrixOperator();
        $x = $mo->array([[1,2,3],[4,5,6]]);
        $this->assertEquals([5,7,9],$mo->la()->reduceSum($x,$axis=0)->toArray());
        $this->assertEquals([6,15],$mo->la()->reduceSum($x,$axis=1)->toArray());

        // with offset
        $x = $mo->array([[[9,9,9],[9,9,9]],[[1,2,3],[4,5,6]]]);
        $x = $x[1];
        $this->assertEquals([5,7,9],$mo->la()->reduceSum($x,$axis=0)->toArray());
        $this->assertEquals([6,15],$mo->la()->reduceSum($x,$axis=1)->toArray());
    }

    public function testArgReduceMax()
    {
        $mo = $this->newMatrixOperator();
        $x = $mo->array([[1,2,3],[4,5,6]]);
        $this->assertEquals([1,1,1],$mo->la()->reduceArgMax($x,$axis=0)->toArray());
        $this->assertEquals([2,2],$mo->la()->reduceArgMax($x,$axis=1)->toArray());

        // with offset
        $x = $mo->array([[[9,9,9],[9,9,9]],[[1,2,3],[4,5,6]]]);
        $x = $x[1];
        $this->assertEquals([1,1,1],$mo->la()->reduceArgMax($x,$axis=0)->toArray());
        $this->assertEquals([2,2],$mo->la()->reduceArgMax($x,$axis=1)->toArray());
    }

    public function testReduceMax()
    {
        $mo = $this->newMatrixOperator();
        $x = $mo->array([[1,2,3],[4,5,6]]);
        $this->assertEquals([4,5,6],$mo->la()->reduceMax($x,$axis=0)->toArray());
        $this->assertEquals([3,6],$mo->la()->reduceMax($x,$axis=1)->toArray());

        // with offset
        $x = $mo->array([[[9,9,9],[9,9,9]],[[1,2,3],[4,5,6]]]);
        $x = $x[1];
        $this->assertEquals([4,5,6],$mo->la()->reduceMax($x,$axis=0)->toArray());
        $this->assertEquals([3,6],$mo->la()->reduceMax($x,$axis=1)->toArray());

    }

    public function testReduceMean()
    {
        $mo = $this->newMatrixOperator();
        $x = $mo->array([[1,2,3],[4,5,6]]);
        $this->assertEquals([2.5,3.5,4.5],$mo->la()->reduceMean($x,$axis=0)->toArray());
        $this->assertEquals([2,5],$mo->la()->reduceMean($x,$axis=1)->toArray());

        // with offset
        $x = $mo->array([[[9,9,9],[9,9,9]],[[1,2,3],[4,5,6]]]);
        $x = $x[1];
        $this->assertEquals([2.5,3.5,4.5],$mo->la()->reduceMean($x,$axis=0)->toArray());
        $this->assertEquals([2,5],$mo->la()->reduceMean($x,$axis=1)->toArray());
    }

    public function testEqualNormal()
    {
        $mo = $this->newMatrixOperator();

        $X = $mo->array([100,10,-1000]);
        $Y = $mo->array([100,-10,-1000]);
        $this->assertEquals([1,0,1],$mo->la()->equal($X,$Y)->toArray());
    }

    public function testSoftmax()
    {
        $mo = $this->newMatrixOperator();

        $x = $mo->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
        ]);
        $y = $mo->la()->softmax($x);
        $this->assertTrue($this->equalTest(0.05801,$y[0][0]));
        $this->assertTrue($this->equalTest(0.09564,$y[0][1]));
        $this->assertTrue($this->equalTest(0.15769,$y[0][2]));
        $this->assertTrue($this->equalTest(0.25999,$y[0][3]));
        $this->assertTrue($this->equalTest(0.42865,$y[0][4]));
    }

    public function testastype()
    {
        $mo = $this->newMatrixOperator();
        $math = $mo->la();

        #### int to any
        $X = $mo->array([-1,0,1,2,3],NDArray::int32);
        $dtype = NDArray::float32;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals(NDArray::float32,$Y->dtype());
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::float64;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::int8;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::int16;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::int32;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::int64;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::bool;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([true,false,true,true,true],$Y->toArray());

        #### float to any ######
        $X = $mo->array([-1,0,1,2,3],NDArray::float32);
        $dtype = NDArray::float32;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::float64;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::int8;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::int16;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::int32;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::int64;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::bool;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([true,false,true,true,true],$Y->toArray());

        #### bool to any ######
        $X = $mo->array([true,false,true,true,true],NDArray::bool);
        $dtype = NDArray::float32;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([1,0,1,1,1],$Y->toArray());

        $dtype = NDArray::float64;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([1,0,1,1,1],$Y->toArray());

        $dtype = NDArray::int8;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([1,0,1,1,1],$Y->toArray());

        $dtype = NDArray::int16;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([1,0,1,1,1],$Y->toArray());

        $dtype = NDArray::int32;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([1,0,1,1,1],$Y->toArray());

        $dtype = NDArray::int64;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([1,0,1,1,1],$Y->toArray());

        $dtype = NDArray::bool;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([true,false,true,true,true],$Y->toArray());

        #### float to unsigned ######
        $X = $mo->array([-1,0,1,2,3],NDArray::float32);
        $dtype = NDArray::uint8;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([255,0,1,2,3],$Y->toArray());

        $dtype = NDArray::uint16;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([65535,0,1,2,3],$Y->toArray());

        $dtype = NDArray::uint32;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([4294967295,0,1,2,3],$Y->toArray());

        $dtype = NDArray::uint64;
        $Y = $math->astype($X, $dtype);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());
    }
    
    public function testIm2col2dNormal()
    {
        $mo = $this->newMatrixOperator();

        $batches = 1;
        $im_h = 4;
        $im_w = 4;
        $channels = 3;
        $kernel_h = 3;
        $kernel_w = 3;
        $stride_h = 1;
        $stride_w = 1;
        $padding = null;
        $channels_first = null;
        $cols_channels_first=null;
        $cols = null;
        
        $images = $mo->arange(
            $batches*
            $im_h*$im_w*
            $channels,
            null,null,
            NDArray::float32
        )->reshape([
            $batches,
            $im_h,
            $im_w,
            $channels
        ]);
        $cols = $mo->la()->im2col(
            $images,
            $filterSize=[
                $kernel_h,$kernel_h],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $cols_channels_first
        );
        $out_h = 2;
        $out_w = 2;
        
        $this->assertEquals(
            [
                $batches,
                $out_h,$out_w,
                $kernel_h,$kernel_w,
                $channels,
            ],
            $cols->shape()
        );
        $this->assertEquals(
        [[
          [
           [[[0,1,2],[3,4,5],[6,7,8]],
            [[12,13,14],[15,16,17],[18,19,20]],
            [[24,25,26],[27,28,29],[30,31,32]],],
           [[[3,4,5],[6,7,8],[9,10,11]],
            [[15,16,17],[18,19,20],[21,22,23]],
            [[27,28,29],[30,31,32],[33,34,35]],],
          ],
          [
           [[[12,13,14],[15,16,17],[18,19,20]],
            [[24,25,26],[27,28,29],[30,31,32]],
            [[36,37,38],[39,40,41],[42,43,44]],],
           [[[15,16,17],[18,19,20],[21,22,23]],
            [[27,28,29],[30,31,32],[33,34,35]],
            [[39,40,41],[42,43,44],[45,46,47]],],
          ],
        ]],
        $cols->toArray()
        );
        
        $newImages = $mo->zerosLike($images);
        $mo->la()->col2im(
            $cols,
            $newImages,
            $filterSize=[
                $kernel_h,$kernel_h],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $cols_channels_first
        );
        
        $this->assertEquals(
            $images->toArray(),
            $newImages->toArray()
        );
    }

    public function testIm2col2dForPool()
    {
        $mo = $this->newMatrixOperator();

        $batches = 1;
        $im_h = 4;
        $im_w = 4;
        $channels = 3;
        $kernel_h = 2;
        $kernel_w = 2;
        $stride_h = 2;
        $stride_w = 2;
        $padding = null;
        $channels_first = null;
        $cols_channels_first=true;
        $cols = null;
        
        $images = $mo->arange(
            $batches*
            $im_h*$im_w*
            $channels,
            null,null,
            NDArray::float32
        )->reshape([
            $batches,
            $im_h,
            $im_w,
            $channels
        ]);
        $cols = $mo->la()->im2col(
            $images,
            $filterSize=[
                $kernel_h,$kernel_h],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $cols_channels_first
        );
        $out_h = 2;
        $out_w = 2;
        return;
        
        $this->assertEquals(
            [
                $batches,
                $out_h,$out_w,
                $channels,
                $kernel_h,$kernel_w,
            ],
            $cols->shape()
        );
        $this->assertEquals(
        [[
          [
           [[[0,3],[12,15]],
            [[1,4],[13,16]],
            [[2,5],[14,17]],],
           [[[6,9],[18,21]],
            [[7,10],[19,22]],
            [[8,11],[20,23]],],
          ],
          [
           [[[24,27],[36,39]],
            [[25,28],[37,40]],
            [[26,29],[38,41]],],
           [[[30,33],[42,45]],
            [[31,34],[43,46]],
            [[32,35],[44,46]],],
          ],
        ]],
        $cols->toArray()
        );
        
        $newImages = $mo->zerosLike($images);
        $mo->la()->col2im(
            $cols,
            $newImages,
            $filterSize=[
                $kernel_h,$kernel_h],
            $strides=[
                $stride_h,$stride_w],
            $padding,
            $channels_first,
            $cols_channels_first
        );
        
        $this->assertEquals(
            $images->toArray(),
            $newImages->toArray()
        );
    }
}
