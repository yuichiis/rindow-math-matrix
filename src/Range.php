<?php
namespace Rindow\Math\Matrix;

use IteratorAggregate;
use Traversable;

/**
 * @implements IteratorAggregate<int, int|float>
 */
class Range implements IteratorAggregate
{
    protected int|float $start;
    protected int|float $limit;
    protected int|float $delta;

    public function __construct(
        int|float $limit,
        int|float $start=null,
        int|float $delta=null
    ) {
        $this->limit = $limit;
        $this->start = $start ?? 0;
        $this->delta = $delta ?? (($limit>=$start)? 1 : -1);
    }

    public function start() : int|float
    {
        return $this->start;
    }

    public function limit() : int|float
    {
        return $this->limit;
    }

    public function delta() : int|float
    {
        return $this->delta;
    }

    public function getIterator() : Traversable
    {
        $index = 0;
        $value = $this->start;
        if($this->delta > 0) {
            while($value < $this->limit) {
                yield $index => $value;
                $index++;
                $value += $this->delta;
            }
        } else {
            while($value > $this->limit) {
                yield $index => $value;
                $index++;
                $value += $this->delta;
            }
        }
    }
}
