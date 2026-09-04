library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

-- 8-bit PWM. Free-running counter; output high while counter < duty.
-- PWM frequency = 12 MHz / 256 = 46.875 kHz (flicker-free).
--   duty = 0   -> 0% on
--   duty = 255 -> 255/256 ~= 99.6% on
entity pwm8 is
    port (
        clk     : in  std_logic;
        duty    : in  std_logic_vector(7 downto 0);
        pwm_out : out std_logic
    );
end pwm8;

architecture rtl of pwm8 is
    signal cnt : unsigned(7 downto 0) := (others => '0');
begin
    process(clk)
    begin
        if rising_edge(clk) then
            cnt <= cnt + 1;
        end if;
    end process;

    pwm_out <= '1' when cnt < unsigned(duty) else '0';
end rtl;
