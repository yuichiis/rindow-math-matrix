<?php
class update
{
    public function testEditGraphScatterAdd()
    {
        //$this->markTestSkipped('Tunning only');
        //return;
        $mo = $this->newMatrixOperator();
        $tunner = new OpenCLMathTunner($mo);
        $times[8][8][16] = 0.6;
        $times[8][8][32] = 0.75;
        $times[8][8][64] = 1.4;
        $times[8][8][128] = 1.4;
        $times[8][8][2048] = 0.7;
        $times[8][16][8] = 0.25;
        $times[8][16][1024] = 0.7;
        $times[8][32][8] = 0.8;
        $times[8][32][16] = 0.28;
        $times[8][32][32] = 0.8;
        $times[8][64][8] = 0.9;
        $times[8][64][16] = 0.4;
        $times[8][64][256] = 0.5;
        $times[8][128][128] = 0.7;
        $times[8][256][8] = 0.5;
        $times[8][256][128] = 0.75;
        $times[16][8][8] = 0.35;
        $times[16][8][16] = 0.95;
        $times[16][8][32] = 0.9;
        $times[16][8][64] = 0.8;
        $times[16][8][128] = 0.8;
        $times[16][8][2048] = 0.7;
        $times[16][16][8] = 0.9;
        $times[16][32][8] = 1.2;
        $times[16][32][16] = 0.85;
        $times[16][64][16] = 1.1;
        $times[16][128][8] = 0.8;
        $times[16][256][8] = 0.8;
        $times[16][256][16] = 0.5;
        $times[32][8][16] = 1.1;
        $times[32][8][32] = 0.4;
        $times[32][8][64] = 0.9;
        $times[32][8][128] = 0.8;
        $times[32][8][1024] = 0.45;
        $times[32][16][8] = 1.3;
        $times[32][64][8] = 0.9;
        $times[32][128][8] = 0.8;
        $times[32][128][16] = 0.7;
        $times[32][256][16] = 1.2;
        $times[64][8][8] = 0.85;
        $times[64][8][64] = 0.9;
        $times[64][8][128] = 1.1;
        $times[64][8][1024] = 0.65;
        $times[64][16][8] = 1.3;
        $times[64][32][8] = 0.7;
        $times[64][32][16] = 0.4;
        $times[64][64][16] = 1.2;
        $times[64][256][16] = 0.7;
        $times[128][8][8] = 1.3;
        $times[128][8][16] = 1.5;
        $times[128][8][1024] = 0.65;
        $times[128][32][16] = 0.6;
        $times[128][512][8] = 0.55;
        $times[256][8][512] = 0.7;
        $times[256][16][8] = 0.4;
        $times[256][32][16] = 0.6;
        $times[256][256][8] = 0.7;
        $times[512][8][8] = 0.8;
        $times[512][8][16] = 0.6;
        $times[512][8][128] = 0.6;
        $times[1024][64][8] = 0.7;
        $times[1024][128][8] = 0.8;
        $times[2048][8][32] = 0.5;
        $times[8192][8][8] = 0.5;
        $times[8192][16][8] = 0.65;
        $times[16384][8][8] = 0.75;
        $tunner->editGraphScatterAdd($mode=2,$times);
        $this->assertTrue(true);
    }
}
