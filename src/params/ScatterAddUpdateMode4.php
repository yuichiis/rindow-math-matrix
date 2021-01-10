<?php
class update
{
    public function testEditGraphScatterAdd()
    {
        //$this->markTestSkipped('Tunning only');
        //return;
        $mo = $this->newMatrixOperator();
        $tunner = new OpenCLMathTunner($mo);
        $times[8][8][8] = 0.3;
        $times[8][128][8] = 0.16;
        $times[8][256][8] = 0.7;
        $times[8][512][8] = 0.4;
        $times[16][8][8] = 0.4;
        $times[16][16][8] = 0.45;
        $times[16][16][16] = 0.45;
        $times[16][32][8] = 0.5;
        $times[16][32][16] = 0.5;
        $times[16][64][8] = 0.7;
        $times[16][64][16] = 0.7;
        $times[16][128][8] = 0.8;
        $times[16][128][16] = 0.8;
        $times[16][512][8] = 1.2;
        $times[16][512][16] = 1.2;
        $times[32][8][8] = 0.6;
        $times[32][16][8] = 0.68;
        $times[32][16][16] = 0.75;
        $times[32][32][8] = 0.8;
        $times[32][32][16] = 0.8;
        $times[32][64][8] = 1.4;
        $times[32][64][16] = 1.4;
        $times[32][128][8] = 0.35;
        $times[32][128][16] = 0.35;
        $times[128][8][8] = 0.22;
        $times[128][64][8] = 0.6;
        $times[128][64][16] = 0.6;
        $times[256][8][8] = 0.46;
        $times[256][16][8] = 0.35;
        $times[256][16][16] = 0.35;
        $times[256][32][8] = 1.2;
        $times[256][32][16] = 1.2;
        $times[512][128][8] = 0.7;
        $times[1024][64][8] = 0.8;
        $times[2048][32][8] = 0.6;
        $times[2048][32][16] = 0.6;

        for($n=8;$n<=1048576;$n<<=1) {
            for($m=8;$m<=1048576;$m<<=1) {
                if(isset($times[$m][$n][8])){
                    for($k=8;$k<=1048576;$k<<=1) {
                        $times[$m][$n][$k] = $times[$m][$n][8];
                    }
                }
            }
        }
        $tunner->editGraphScatterAdd($mode=4,$times);
        $this->assertTrue(true);
    }
}
