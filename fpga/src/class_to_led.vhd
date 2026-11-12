library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

-- Maps a keyword class ID to LED brightness channels.
-- Output channels: LD1, LD2, R, G, B.
entity class_to_led is
    port (
        class_id : in  unsigned(3 downto 0);
        valid    : in  std_logic;
        ch0      : out std_logic_vector(7 downto 0);
        ch1      : out std_logic_vector(7 downto 0);
        red      : out std_logic_vector(7 downto 0);
        green    : out std_logic_vector(7 downto 0);
        blue     : out std_logic_vector(7 downto 0)
    );
end class_to_led;

architecture rtl of class_to_led is
begin
    process(class_id, valid)
    begin
        ch0   <= x"00";
        ch1   <= x"00";
        red   <= x"00";
        green <= x"00";
        blue  <= x"00";

        if valid = '1' then
            case to_integer(class_id) is
                when 2 => green <= x"FE"; ch0 <= x"80"; -- yes
                when 3 => red   <= x"FE"; ch1 <= x"80"; -- no
                when 4 => blue  <= x"FE";                -- up
                when 5 => green <= x"FE"; blue <= x"FE"; -- down
                when 6 => red   <= x"80"; blue <= x"FE"; -- left
                when 7 => red   <= x"FE"; green <= x"80"; -- right
                when 8 => green <= x"FE"; red <= x"80";  -- on
                when 9 => blue  <= x"80";                -- off
                when 10 => red  <= x"FE"; ch0 <= x"FE"; ch1 <= x"FE"; -- stop
                when 11 => green <= x"FE"; blue <= x"80"; -- go
                when others => null; -- silence or unknown
            end case;
        end if;
    end process;
end rtl;
