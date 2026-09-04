library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

-- Phase 2 starter for the FPGA keyword-spotting MLP.
-- The full implementation will load exported .mem weights and compute:
-- 490 -> 128 -> 64 -> 12, int8 weights/activations, int32 accumulators.
entity kws_mlp is
    port (
        clk       : in  std_logic;
        reset     : in  std_logic;
        start     : in  std_logic;
        busy      : out std_logic;
        done      : out std_logic;
        class_id  : out unsigned(3 downto 0);
        class_ok  : out std_logic
    );
end kws_mlp;

architecture rtl of kws_mlp is
    signal done_r : std_logic := '0';
begin
    process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' then
                done_r <= '0';
            else
                done_r <= start;
            end if;
        end if;
    end process;

    busy     <= '0';
    done     <= done_r;
    class_id <= to_unsigned(0, 4); -- silence until weights/datapath are added
    class_ok <= '0';
end rtl;
