library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

-- Phase 2 starter: stores the latest 49 frames x 10 int8 MFCC coefficients.
-- The flattened output presents oldest frame first after the buffer fills.
entity mfcc_ring_buffer is
    port (
        clk          : in  std_logic;
        reset        : in  std_logic;
        frame_valid  : in  std_logic;
        coeff_index  : in  unsigned(3 downto 0);
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
            elsif frame_valid = '1' and coeff_index = to_unsigned(9, 4) then
                if frame_count < to_unsigned(49, 6) then
                    frame_count <= frame_count + 1;
                end if;
            end if;
        end if;
    end process;

    window_ready <= '1' when frame_count = to_unsigned(49, 6) else '0';
    write_count  <= frame_count;
end rtl;
