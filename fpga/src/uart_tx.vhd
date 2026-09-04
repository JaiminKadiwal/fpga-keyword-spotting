library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

-- Simple 8N1 UART transmitter (mirror of uart_rx).
-- Idle-high line, one start bit, 8 data bits LSB-first, one stop bit.
-- Assert tx_start for one clock (while tx_busy = '0') to latch tx_data and
-- begin sending. Default divisor is for 12 MHz / 115200.
entity uart_tx is
    generic (
        CLKS_PER_BIT : integer := 104   -- round(12_000_000 / 115200)
    );
    port (
        clk      : in  std_logic;
        tx_start : in  std_logic;                    -- one-clk request to send tx_data
        tx_data  : in  std_logic_vector(7 downto 0);
        tx       : out std_logic;                    -- serial output, idle high
        tx_busy  : out std_logic                     -- high while a byte is in flight
    );
end uart_tx;

architecture rtl of uart_tx is
    type state_t is (S_IDLE, S_START, S_DATA, S_STOP);
    signal state   : state_t := S_IDLE;
    signal clk_cnt : integer range 0 to CLKS_PER_BIT-1 := 0;
    signal bit_idx : integer range 0 to 7 := 0;
    signal shreg   : std_logic_vector(7 downto 0) := (others => '0');
    signal tx_r    : std_logic := '1';               -- idle high
    signal busy_r  : std_logic := '0';
begin
    tx      <= tx_r;
    tx_busy <= busy_r;

    process(clk)
    begin
        if rising_edge(clk) then
            case state is
                when S_IDLE =>
                    tx_r    <= '1';
                    clk_cnt <= 0;
                    bit_idx <= 0;
                    if tx_start = '1' then
                        shreg  <= tx_data;
                        busy_r <= '1';
                        tx_r   <= '0';                -- start bit
                        state  <= S_START;
                    else
                        busy_r <= '0';
                    end if;

                when S_START =>                       -- hold start bit one bit period
                    if clk_cnt < CLKS_PER_BIT-1 then
                        clk_cnt <= clk_cnt + 1;
                    else
                        clk_cnt <= 0;
                        tx_r    <= shreg(0);          -- first data bit (LSB)
                        state   <= S_DATA;
                    end if;

                when S_DATA =>                        -- one full bit period per data bit
                    if clk_cnt < CLKS_PER_BIT-1 then
                        clk_cnt <= clk_cnt + 1;
                    else
                        clk_cnt <= 0;
                        if bit_idx < 7 then
                            bit_idx <= bit_idx + 1;
                            tx_r    <= shreg(bit_idx + 1);
                        else
                            bit_idx <= 0;
                            tx_r    <= '1';           -- stop bit
                            state   <= S_STOP;
                        end if;
                    end if;

                when S_STOP =>                        -- hold stop bit one bit period
                    if clk_cnt < CLKS_PER_BIT-1 then
                        clk_cnt <= clk_cnt + 1;
                    else
                        clk_cnt <= 0;
                        busy_r  <= '0';
                        state   <= S_IDLE;
                    end if;

                when others =>
                    state <= S_IDLE;
            end case;
        end if;
    end process;
end rtl;
