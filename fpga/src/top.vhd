library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

-- Sound-reactive LED visualizer for the Digilent Cmod A7 (xc7a35t).
--
-- The laptop mic is captured on the PC; a Python script computes audio
-- features and streams them over the USB-UART bridge as frames:
--
--     0xFF  ch0  ch1  ch2  ch3  ch4        (each data byte 0..254)
--
--   ch0 -> LD1  (green user LED)  : low/mid "body" level
--   ch1 -> LD2  (green user LED)  : treble "sparkle" level
--   ch2 -> RGB red               : bass energy
--   ch3 -> RGB green             : mid energy
--   ch4 -> RGB blue              : treble energy
--
-- 0xFF is the frame sync and never appears in data, so the parser
-- re-synchronises automatically. If no frame arrives for ~0.4 s the
-- design falls back to a slow rainbow "idle/listening" animation.
--
-- Board polarity note: LD1/LD2 are active-HIGH; the RGB LED LD0 is
-- common-anode / active-LOW, so its PWM outputs are inverted. RGB duty
-- is capped at ~75% (bright drops, still short of full glare).
--
-- Buttons: BTN0 = lamp-test (all LEDs full, no PC needed);
--          BTN1 = force the idle rainbow (demo without audio).
entity top is
    port (
        sysclk       : in  std_logic;                    -- 12 MHz, L17
        uart_txd_in  : in  std_logic;                    -- PC -> FPGA serial, J17
        uart_rxd_out : out std_logic;                    -- FPGA -> PC serial, J18 (idle)
        btn          : in  std_logic_vector(1 downto 0);
        led          : out std_logic_vector(1 downto 0); -- LD1, LD2 (active high)
        led0_r       : out std_logic;                    -- RGB LD0 red   (active low)
        led0_g       : out std_logic;                    -- RGB LD0 green (active low)
        led0_b       : out std_logic                     -- RGB LD0 blue  (active low)
    );
end top;

architecture Behavioral of top is

    component uart_rx is
        generic ( CLKS_PER_BIT : integer := 104 );
        port (
            clk        : in  std_logic;
            rx         : in  std_logic;
            data_out   : out std_logic_vector(7 downto 0);
            data_valid : out std_logic
        );
    end component;

    component pwm8 is
        port (
            clk     : in  std_logic;
            duty    : in  std_logic_vector(7 downto 0);
            pwm_out : out std_logic
        );
    end component;

    type byte_array is array (0 to 4) of std_logic_vector(7 downto 0);

    -- UART / frame parser
    signal rx_data     : std_logic_vector(7 downto 0);
    signal rx_dv       : std_logic;
    signal collecting  : std_logic := '0';
    signal idx         : integer range 0 to 4 := 0;
    signal buf         : byte_array := (others => (others => '0'));
    signal ch          : byte_array := (others => (others => '0'));
    signal frame_latch : std_logic := '0';

    -- watchdog: "audio is live" while frames keep arriving
    constant WD_ACTIVE : unsigned(23 downto 0) := to_unsigned(5_000_000, 24); -- ~0.42 s
    signal   wd_cnt    : unsigned(23 downto 0) := (others => '1');
    signal   active    : std_logic;

    -- idle rainbow animation
    signal phase       : unsigned(23 downto 0) := (others => '0');
    signal r_idle      : std_logic_vector(7 downto 0) := (others => '0');
    signal g_idle      : std_logic_vector(7 downto 0) := (others => '0');
    signal b_idle      : std_logic_vector(7 downto 0) := (others => '0');
    signal green_idle  : std_logic_vector(7 downto 0) := (others => '0');

    -- button synchroniser
    signal btn_s0, btn_s1 : std_logic_vector(1 downto 0) := (others => '0');

    -- brightness selected into the PWM channels
    signal duty_ld1, duty_ld2         : std_logic_vector(7 downto 0);
    signal duty_r,   duty_g,   duty_b : std_logic_vector(7 downto 0);
    signal rgb_r_h,  rgb_g_h,  rgb_b_h : std_logic_vector(7 downto 0); -- capped to 50%

    signal g0_pwm, g1_pwm, r_pwm, g_pwm, b_pwm : std_logic;

    -- triangle wave: 0->254->0 as v sweeps 0..255 (for a hue-like ramp)
    function tri(v : unsigned(7 downto 0)) return unsigned is
    begin
        if v < 128 then
            return v(6 downto 0) & '0';          -- v * 2
        else
            return (not v(6 downto 0)) & '0';    -- (255 - v) * 2
        end if;
    end function;

begin

    uart_rxd_out <= '1';                          -- keep bridge RX line idle-high

    ------------------------------------------------------------------
    -- UART receiver
    ------------------------------------------------------------------
    u_rx : uart_rx
        generic map ( CLKS_PER_BIT => 104 )
        port map (
            clk        => sysclk,
            rx         => uart_txd_in,
            data_out   => rx_data,
            data_valid => rx_dv
        );

    ------------------------------------------------------------------
    -- Frame parser: 0xFF sync, then 5 channel bytes
    ------------------------------------------------------------------
    process(sysclk)
    begin
        if rising_edge(sysclk) then
            frame_latch <= '0';
            if rx_dv = '1' then
                if rx_data = x"FF" then
                    idx        <= 0;
                    collecting <= '1';
                elsif collecting = '1' then
                    if idx = 4 then
                        ch(0)       <= buf(0);
                        ch(1)       <= buf(1);
                        ch(2)       <= buf(2);
                        ch(3)       <= buf(3);
                        ch(4)       <= rx_data;
                        collecting  <= '0';
                        frame_latch <= '1';
                    else
                        buf(idx) <= rx_data;
                        idx      <= idx + 1;
                    end if;
                end if;
            end if;
        end if;
    end process;

    ------------------------------------------------------------------
    -- Watchdog: active while frames keep coming
    ------------------------------------------------------------------
    process(sysclk)
    begin
        if rising_edge(sysclk) then
            if frame_latch = '1' then
                wd_cnt <= (others => '0');
            elsif wd_cnt < to_unsigned(16777215, 24) then
                wd_cnt <= wd_cnt + 1;
            end if;
        end if;
    end process;
    active <= '1' when wd_cnt < WD_ACTIVE else '0';

    ------------------------------------------------------------------
    -- Idle rainbow generator
    ------------------------------------------------------------------
    process(sysclk)
        variable h  : unsigned(7 downto 0);
        variable tr : unsigned(7 downto 0);
    begin
        if rising_edge(sysclk) then
            phase <= phase + 1;
            h  := phase(23 downto 16);              -- full cycle ~1.4 s
            tr := tri(h);
            r_idle     <= std_logic_vector(tri(h));
            g_idle     <= std_logic_vector(tri(h + 85));
            b_idle     <= std_logic_vector(tri(h + 170));
            green_idle <= std_logic_vector("00" & tr(7 downto 2)); -- dim breathing
        end if;
    end process;

    ------------------------------------------------------------------
    -- Button synchroniser
    ------------------------------------------------------------------
    process(sysclk)
    begin
        if rising_edge(sysclk) then
            btn_s0 <= btn;
            btn_s1 <= btn_s0;
        end if;
    end process;

    ------------------------------------------------------------------
    -- Source select: lamp-test / forced-demo / audio / idle
    ------------------------------------------------------------------
    process(btn_s1, active, ch, r_idle, g_idle, b_idle, green_idle)
    begin
        if btn_s1(0) = '1' then                     -- lamp test
            duty_ld1 <= x"FF"; duty_ld2 <= x"FF";
            duty_r   <= x"FF"; duty_g   <= x"FF"; duty_b <= x"FF";
        elsif btn_s1(1) = '1' then                  -- forced idle rainbow
            duty_ld1 <= green_idle; duty_ld2 <= green_idle;
            duty_r   <= r_idle; duty_g <= g_idle; duty_b <= b_idle;
        elsif active = '1' then                     -- live audio
            duty_ld1 <= ch(0); duty_ld2 <= ch(1);
            duty_r   <= ch(2); duty_g <= ch(3); duty_b <= ch(4);
        else                                        -- no data: idle rainbow
            duty_ld1 <= green_idle; duty_ld2 <= green_idle;
            duty_r   <= r_idle; duty_g <= g_idle; duty_b <= b_idle;
        end if;
    end process;

    -- Cap RGB brightness at ~75% (duty*3/4 = duty/2 + duty/4) for a brighter drop.
    -- Shift-and-add keeps it to 8-bit adds with no overflow (max 127+63 = 190).
    rgb_r_h <= std_logic_vector(shift_right(unsigned(duty_r), 1) + shift_right(unsigned(duty_r), 2));
    rgb_g_h <= std_logic_vector(shift_right(unsigned(duty_g), 1) + shift_right(unsigned(duty_g), 2));
    rgb_b_h <= std_logic_vector(shift_right(unsigned(duty_b), 1) + shift_right(unsigned(duty_b), 2));

    ------------------------------------------------------------------
    -- PWM channels
    ------------------------------------------------------------------
    pwm_g0 : pwm8 port map ( clk => sysclk, duty => duty_ld1, pwm_out => g0_pwm );
    pwm_g1 : pwm8 port map ( clk => sysclk, duty => duty_ld2, pwm_out => g1_pwm );
    pwm_r  : pwm8 port map ( clk => sysclk, duty => rgb_r_h,  pwm_out => r_pwm  );
    pwm_g  : pwm8 port map ( clk => sysclk, duty => rgb_g_h,  pwm_out => g_pwm  );
    pwm_b  : pwm8 port map ( clk => sysclk, duty => rgb_b_h,  pwm_out => b_pwm  );

    -- Green LEDs: active high.  RGB LED: active low (invert).
    led(0) <= g0_pwm;
    led(1) <= g1_pwm;
    led0_r <= not r_pwm;
    led0_g <= not g_pwm;
    led0_b <= not b_pwm;

end Behavioral;
