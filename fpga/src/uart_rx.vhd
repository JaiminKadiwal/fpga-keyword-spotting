library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

-- Simple 8N1 UART receiver.
-- Samples at the middle of each bit. Default divisor is for 12 MHz / 115200.
entity uart_rx is
    generic (
        CLKS_PER_BIT : integer := 104   -- round(12_000_000 / 115200)
    );
    port (
        clk        : in  std_logic;
        rx         : in  std_logic;                    -- asynchronous serial input
        data_out   : out std_logic_vector(7 downto 0);
        data_valid : out std_logic                     -- one-clk strobe when a byte is ready
    );
end uart_rx;

architecture rtl of uart_rx is
    type state_t is (S_IDLE, S_START, S_DATA, S_STOP, S_DONE);
    signal state    : state_t := S_IDLE;
    signal rx_sync0 : std_logic := '1';
    signal rx_sync1 : std_logic := '1';               -- metastability double-flop
    signal clk_cnt  : integer range 0 to CLKS_PER_BIT-1 := 0;
    signal bit_idx  : integer range 0 to 7 := 0;
    signal rx_byte  : std_logic_vector(7 downto 0) := (others => '0');
    signal dv       : std_logic := '0';
begin
    process(clk)
    begin
        if rising_edge(clk) then
            rx_sync0 <= rx;
            rx_sync1 <= rx_sync0;
        end if;
    end process;

    data_out   <= rx_byte;
    data_valid <= dv;

    process(clk)
    begin
        if rising_edge(clk) then
            dv <= '0';                                 -- default: no strobe
            case state is
                when S_IDLE =>
                    clk_cnt <= 0;
                    bit_idx <= 0;
                    if rx_sync1 = '0' then             -- falling edge = start bit
                        state <= S_START;
                    end if;

                when S_START =>                        -- confirm start at mid-bit
                    if clk_cnt = (CLKS_PER_BIT-1)/2 then
                        if rx_sync1 = '0' then
                            clk_cnt <= 0;
                            state   <= S_DATA;
                        else
                            state <= S_IDLE;           -- glitch, not a real start
                        end if;
                    else
                        clk_cnt <= clk_cnt + 1;
                    end if;

                when S_DATA =>                         -- one full bit period per data bit
                    if clk_cnt < CLKS_PER_BIT-1 then
                        clk_cnt <= clk_cnt + 1;
                    else
                        clk_cnt          <= 0;
                        rx_byte(bit_idx) <= rx_sync1;  -- LSB first
                        if bit_idx < 7 then
                            bit_idx <= bit_idx + 1;
                        else
                            bit_idx <= 0;
                            state   <= S_STOP;
                        end if;
                    end if;

                when S_STOP =>
                    if clk_cnt < CLKS_PER_BIT-1 then
                        clk_cnt <= clk_cnt + 1;
                    else
                        clk_cnt <= 0;
                        dv      <= '1';                -- byte complete
                        state   <= S_DONE;
                    end if;

                when S_DONE =>
                    state <= S_IDLE;

                when others =>
                    state <= S_IDLE;
            end case;
        end if;
    end process;
end rtl;
