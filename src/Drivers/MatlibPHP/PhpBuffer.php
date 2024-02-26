<?php
namespace Rindow\Math\Matrix\Drivers\MatlibPHP;

use Interop\Polite\Math\Matrix\Buffer as BufferInterface;
use Interop\Polite\Math\Matrix\NDArray;
use TypeError;
use InvalidArgumentException;
use OutOfRangeException;
use LogicException;
use FFI;
use SplFixedArray;

class PhpBuffer extends SplFixedArray implements BufferInterface
{
    protected static $typeString = [
        NDArray::bool    => 'uint8_t',
        NDArray::int8    => 'int8_t',
        NDArray::int16   => 'int16_t',
        NDArray::int32   => 'int32_t',
        NDArray::int64   => 'int64_t',
        NDArray::uint8   => 'uint8_t',
        NDArray::uint16  => 'uint16_t',
        NDArray::uint32  => 'uint32_t',
        NDArray::uint64  => 'uint64_t',
        //NDArray::float8  => 'N/A',
        //NDArray::float16 => 'N/A',
        NDArray::float32 => 'float',
        NDArray::float64 => 'double',
    ];

    protected int $dtype;

    public function __construct(int $size, int $dtype)
    {
        if(!isset(self::$typeString[$dtype])) {
            throw new InvalidArgumentException("Invalid data type");
        }
        $this->dtype = $dtype;
        parent::__construct($size);
    }

    public static function fromArray(array $array, bool $preserveKeys = true) : SplFixedArray
    {
        throw new LogicException("Unsupported operation");
    }

    public static function fromArrayWithDtype(array $array, int $dtype) : BufferInterface
    {
        $a = new self(count($array), $dtype);
        foreach($array as $i => $v) {
            if(!is_int($i)) {
                throw new InvalidArgumentException("array must contain only positive integer keys");
            }
            $a[$i] = $v;
        }
        return $a;
    }

    public function dtype() : int
    {
        return $this->dtype;
    }

}
