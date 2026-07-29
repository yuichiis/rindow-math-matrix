<?php
/**
 * PCG32 - 拡張非依存の純PHP版
 *
 * GMP等のPHP拡張を一切使わず、64bit整数を「16bitの limb を4つ並べた配列」
 * (リトルエンディアン: $limbs[0] が下位16bit, $limbs[3] が上位16bit)として
 * 表現し、加算・乗算・シフトを自前の桁上げ処理で実装しています。
 *
 * Pcg32.php（GMP版）およびpcg32.c（Cネイティブ版）と同一アルゴリズム・
 * 同一シードで完全に同じ乱数列を生成します。実行速度はネイティブ版・GMP版
 * より大幅に遅いため、拡張が使えない環境でのフォールバック用途を想定しています。
 */
namespace Rindow\Math\Matrix\Drivers\MatlibPHP;

class PhpPcg32
{
    /** 6364136223846793005 = 0x5851F42D4C957F2D を16bit limb 4つに分解 */
    private const MULT = [0x7F2D, 0x4C95, 0xF42D, 0x5851];

    /** @var int[] 4要素、各0-65535（下位から上位へ） */
    private array $state;

    /** @var int[] 4要素、各0-65535 */
    private array $inc;

    public function __construct(int $seed = 0, int $sequence = 1)
    {
        // inc = (sequence << 1) | 1 （必ず奇数にする）
        $this->inc = self::fromInt(($sequence << 1) | 1);

        $this->setSeed($seed);
    }

    // ---- 64bit演算ヘルパー（すべてlimb配列を返す） ----

    private static function fromInt(int $v): array
    {
        return [
            $v & 0xFFFF,
            ($v >> 16) & 0xFFFF,
            ($v >> 32) & 0xFFFF,
            ($v >> 48) & 0xFFFF,
        ];
    }

    /** mod 2^64 の加算 */
    private static function add64(array $a, array $b): array
    {
        $carry = 0;
        $r = [0, 0, 0, 0];
        for ($i = 0; $i < 4; $i++) {
            $sum = $a[$i] + $b[$i] + $carry;
            $r[$i] = $sum & 0xFFFF;
            $carry = $sum >> 16;
        }
        return $r; // 最終桁上げは mod 2^64 なので捨てる
    }

    /** mod 2^64 の乗算（16bit limb同士の筆算） */
    private static function mul64(array $a, array $b): array
    {
        $tmp = [0, 0, 0, 0, 0, 0, 0, 0];
        for ($i = 0; $i < 4; $i++) {
            $carry = 0;
            for ($j = 0; $j < 4; $j++) {
                $pos = $i + $j;
                if ($pos > 7) {
                    continue;
                }
                $prod = $a[$i] * $b[$j] + $tmp[$pos] + $carry;
                $tmp[$pos] = $prod & 0xFFFF;
                $carry = intdiv($prod, 0x10000);
            }
            $k = $i + 4;
            while ($carry > 0 && $k < 8) {
                $prod = $tmp[$k] + $carry;
                $tmp[$k] = $prod & 0xFFFF;
                $carry = intdiv($prod, 0x10000);
                $k++;
            }
        }
        return [$tmp[0], $tmp[1], $tmp[2], $tmp[3]]; // 上位4 limb は mod 2^64 で切り捨て
    }

    /** 各limbのXOR */
    private static function xor64(array $a, array $b): array
    {
        return [$a[0] ^ $b[0], $a[1] ^ $b[1], $a[2] ^ $b[2], $a[3] ^ $b[3]];
    }

    /** 論理右シフト（符号なし扱い、$n は0〜63） */
    private static function shr64(array $a, int $n): array
    {
        if ($n <= 0) {
            return $a;
        }
        if ($n >= 64) {
            return [0, 0, 0, 0];
        }
        $limbShift = intdiv($n, 16);
        $bitShift = $n % 16;
        $r = [0, 0, 0, 0];
        for ($i = 0; $i < 4; $i++) {
            $lowIdx = $i + $limbShift;
            $highIdx = $lowIdx + 1;
            $low = ($lowIdx <= 3) ? $a[$lowIdx] : 0;
            $high = ($highIdx <= 3) ? $a[$highIdx] : 0;
            $val = ($bitShift === 0)
                ? $low
                : (($low >> $bitShift) | (($high << (16 - $bitShift)) & 0xFFFF));
            $r[$i] = $val & 0xFFFF;
        }
        return $r;
    }

    /** limb配列の下位32bitをPHPのintとして取り出す */
    private static function toUint32(array $a): int
    {
        return $a[0] | ($a[1] << 16);
    }

    private function step(): void
    {
        $this->state = self::add64(self::mul64($this->state, self::MULT), $this->inc);
    }

    // ---- 公開API（Pcg32.php / pcg32.c と同じインターフェース） ----

    public function setSeed(int $seed): void
    {
        $this->state = [0, 0, 0, 0];
        $this->step();
        $this->state = self::add64($this->state, self::fromInt($seed));
        $this->step();
    }

    public function randUint32(): int
    {
        $oldstate = $this->state;
        $this->step();

        $xorshifted64 = self::shr64(
            self::xor64(self::shr64($oldstate, 18), $oldstate),
            27
        );
        $xorshifted = self::toUint32($xorshifted64);

        $rot = self::toUint32(self::shr64($oldstate, 59)) & 31;

        return (($xorshifted >> $rot) | ($xorshifted << ((-$rot) & 31))) & 0xFFFFFFFF;
    }

    /**
     * Uint32(0 〜 4294967295)をInt32(-2147483648 〜 2147483647)へ変換する。
     * ビットパターンはそのまま、最上位ビット(bit31)を符号ビットとして
     * 再解釈する（2の補数表現）。
     */
    public static function uint32ToInt32(int $u): int
    {
        $u &= 0xFFFFFFFF;
        return ($u >= 0x80000000) ? $u - 0x100000000 : $u;
    }
 
    /**
     * 次の乱数を符号あり32bit整数（-2147483648 〜 2147483647）として返す。
     */
    public function nextInt32(): int
    {
        return self::uint32ToInt32($this->nextUint32());
    }

    public function nextFloat(): float
    {
        return $this->randUint32() / 4294967296.0;
    }

    public function nextInt(int $min, int $max): int
    {
        $range = $max - $min + 1;
        if ($range <= 0) {
            throw new \InvalidArgumentException('max must be >= min');
        }
        $threshold = (0x100000000 - $range) % $range;
        do {
            $r = $this->randUint32();
        } while ($r < $threshold);

        return $min + ($r % $range);
    }

    /** @return float[] */
    public function fillUniform(int $size, float $low = 0.0, float $high = 1.0): array
    {
        $out = [];
        $scale = $high - $low;
        for ($i = 0; $i < $size; $i++) {
            $out[] = $low + $this->nextFloat() * $scale;
        }
        return $out;
    }

    /** @return array{state:int[], inc:int[]} */
    public function getState(): array
    {
        return ['state' => $this->state, 'inc' => $this->inc];
    }

    /** @param array{state:int[], inc:int[]} $stateData */
    public function setState(array $stateData): void
    {
        $this->state = $stateData['state'];
        $this->inc = $stateData['inc'];
    }
}