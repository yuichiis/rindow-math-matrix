<?php
namespace Rindow\Math\Matrix\Drivers\CLBlastExt;

use Rindow\CLBlast\Blas;
use Rindow\CLBlast\Math;
use Rindow\Math\Matrix\Drivers\AbstractDriver;
use Rindow\Math\Matrix\Drivers\Service;

class CLBlastFactory extends AbstractDriver
{
    protected string $LOWEST_VERSION = '0.1.2';
    protected string $OVER_VERSION   = '0.2.0';

    protected string $extName = 'rindow_clblast';

    public function Blas(object $queue,Service $service) : object
    {
        $this->assertVersion();
        return new Blas();
    }

    public function Math(object $queue,Service $service) : object
    {
        $this->assertVersion();
        return new Math();
    }
}
