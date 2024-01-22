<?php
namespace Rindow\Math\Matrix\Drivers;

use Rindow\Math\Buffer\FFI\BufferFactory;
use Rindow\OpenBLAS\FFI\OpenBLASFactory;
use Rindow\Matlib\FFI\MatlibFactory;
use Rindow\OpenCL\FFI\OpenCLFactory;
use Rindow\CLBlast\FFI\CLBlastFactory;
use Rindow\Math\Matrix\Drivers\MatlibCL\MatlibCLFactory;

class MatlibPhp extends AbstractMatlibService
{
    protected $name = 'matlib_php';

    public function __construct(
        object $bufferFactory=null,
        object $mathFactory=null,
        object $openblasFactory=null,
        object $openclFactory=null,
        object $clblastFactory=null,
        object $blasCLFactory=null,
        object $mathCLFactory=null,
        object $bufferCLFactory=null,
        )
    {
        //$bufferFactory = $bufferFactory ?? new BufferFactory();
//
        //$openblasFactory = $openblasFactory ?? new OpenBLASFactory();
//
        //$mathFactory = $mathFactory ?? new MatlibFactory();
//
        //$openclFactory = $openclFactory ?? new OpenCLFactory();
        //$bufferCLFactory = $bufferCLFactory ?? $openclFactory;
//
        //$clblastFactory = $clblastFactory ?? new CLBlastFactory();
        //$blasCLFactory = $blasCLFactory ?? $clblastFactory;
//
        //$mathCLFactory = $mathCLFactory ?? new MatlibCLFactory();

        parent::__construct();
        //    bufferFactory:$bufferFactory,
        //    openblasFactory:$openblasFactory,
        //    mathFactory:$mathFactory,
        //    openclFactory:$openclFactory,
        //    clblastFactory:$clblastFactory,
        //    blasCLFactory:$blasCLFactory,
        //    mathCLFactory:$mathCLFactory,
        //    bufferCLFactory:$bufferCLFactory,
        //);
    }
}