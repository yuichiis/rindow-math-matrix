<?php
namespace Rindow\Math\Matrix\Drivers\OpenBLASExt;

use Interop\Polite\Math\Matrix\Buffer;
use Rindow\OpenBLAS\Blas;
use Rindow\OpenBLAS\Lapack;
use Rindow\OpenBLAS\Math;
use Rindow\Math\Matrix\Drivers\AbstractDriver;

class OpenBLASFactory extends AbstractDriver
{
    protected string $LOWEST_VERSION = '0.4.0';
    protected string $OVER_VERSION   = '0.5.0';
    protected string $extName = 'rindow_openblas';

    public function Blas() : object
    {
        $this->assertVersion();
        return new Blas();
    }

    public function Lapack() : object
    {
        $this->assertVersion();
        return new Lapack();
    }

    public function Math() : object
    {
        $this->assertVersion();
        return new Math();
    }

    public function Buffer(
        int $size, int $dtype
        ) : Buffer
    {
        return new OpenBlasBuffer($size, $dtype);
    }
}
