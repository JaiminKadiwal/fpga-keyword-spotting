library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

-- Phase 2 starter: stores the latest 49 frames x 20 int8 coefficients
-- (10 MFCC + 10 delta). The flattened output presents oldest frame first
-- after the buffer fills. Coefficient index runs 0..19 per frame.
entity mfcc_ring_buffer is
    port (
        clk          : in  std_logic;
        reset        : in  std_logic;
        frame_valid  : in  std_logic;
        coeff_index  : in  unsigned(4 downto 0);
        coeff_value  : in  signed(7 downto 0);
        window_ready : out std_logic;
        write_count  : out unsigned(5 downto 0)
    );
end mfcc_ring_buffer;

architecture rtl of mfcc_ring_buffer is
    signal frame_count : unsigned(5 downto 0) := (others => '0');
begin
    process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' then
                frame_count <= (others => '0');
            elsif frame_valid = '1' and coeff_index = to_unsigned(19, 5) then
                if frame_count < to_unsigned(49, 6) then
                    frame_count <= frame_count + 1;
                end if;
            end if;
        end if;
    end process;

    window_ready <= '1' when frame_count = to_unsigned(49, 6) else '0';
    write_count  <= frame_count;
end rtl;
