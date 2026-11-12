library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

-- Phase 2C: on-FPGA keyword-spotting inference driving the LEDs.
--
-- The laptop computes MFCC+delta features and streams them over the USB-UART
-- bridge as fixed-length MFCC frames (docs/uart_protocol.md):
--
--     0xFE  frame_index  c0..c9  d0..d9  checksum     (23 bytes)
--
--   checksum = (frame_index + sum(20 coeffs)) & 0xFF
--
-- Coefficients are int8 (two's-complement bytes) and MAY equal 0xFE/0xFF, so
-- the parser uses counted framing + checksum validation, not resync-on-sync.
-- Each accepted frame writes its 20 coefficients into the kws_mlp feature
-- memory at addr = frame_count*20 + coeff_index (oldest frame first, matching
-- docs/fixed_point_contract.md). After 49 frames the parser pulses kws_mlp
-- start; on done the predicted class is latched, mapped to LED colors via
-- class_to_led, and echoed back to the PC as a reply frame:
--
--     0xCA  class_id  (0xCA + class_id) & 0xFF        (3 bytes)
--
-- Timing: kws_mlp only accepts feature writes in its S_IDLE state and inference
-- takes ~2.7e5 cycles (~22 ms at 12 MHz). The PC waits for the reply frame
-- before streaming the next window, so windows never overlap and feature writes
-- never collide with an in-flight inference. The parser also gates writes on
-- mlp_busy = '0' as a belt-and-suspenders guard.
--
-- Board polarity note: LD1/LD2 are active-HIGH; the RGB LED LD0 is
-- common-anode / active-LOW, so its PWM outputs are inverted. RGB duty is
-- capped at ~75%.
--
-- Buttons: BTN0 = lamp-test (all LEDs full, no PC needed);
--          BTN1 = force the idle rainbow (demo without audio).
entity top is
    generic (
        -- Weight/bias .mem paths for kws_mlp. Defaults are repo-relative; the
        -- build script (fpga/vivado/build.tcl) overrides them with absolute
        -- paths so the ROM init files resolve during synthesis elaboration.
        W1_FILE : string := "models/exported/weights_l1.mem";
        B1_FILE : string := "models/exported/bias_l1.mem";
        W2_FILE : string := "models/exported/weights_l2.mem";
        B2_FILE : string := "models/exported/bias_l2.mem";
        W3_FILE : string := "models/exported/weights_l3.mem";
        B3_FILE : string := "models/exported/bias_l3.mem"
    );
    port (
        sysclk       : in  std_logic;                    -- 12 MHz, L17
        uart_txd_in  : in  std_logic;                    -- PC -> FPGA serial, J17
        uart_rxd_out : out std_logic;                    -- FPGA -> PC serial, J18
        btn          : in  std_logic_vector(1 downto 0);
        led          : out std_logic_vector(1 downto 0); -- LD1, LD2 (active high)
        led0_r       : out std_logic;                    -- RGB LD0 red   (active low)
        led0_g       : out std_logic;                    -- RGB LD0 green (active low)
        led0_b       : out std_logic                     -- RGB LD0 blue  (active low)
    );
end top;

architecture Behavioral of top is

    constant CLKS_PER_BIT : integer := 104;              -- 12 MHz / 115200

    component uart_rx is
        generic ( CLKS_PER_BIT : integer := 104 );
        port (
            clk        : in  std_logic;
            rx         : in  std_logic;
            data_out   : out std_logic_vector(7 downto 0);
            data_valid : out std_logic
        );
    end component;

    component uart_tx is
        generic ( CLKS_PER_BIT : integer := 104 );
        port (
            clk      : in  std_logic;
            tx_start : in  std_logic;
            tx_data  : in  std_logic_vector(7 downto 0);
            tx       : out std_logic;
            tx_busy  : out std_logic
        );
    end component;

    component kws_mlp is
        generic (
            W1_FILE : string := "models/exported/weights_l1.mem";
            B1_FILE : string := "models/exported/bias_l1.mem";
            W2_FILE : string := "models/exported/weights_l2.mem";
            B2_FILE : string := "models/exported/bias_l2.mem";
            W3_FILE : string := "models/exported/weights_l3.mem";
            B3_FILE : string := "models/exported/bias_l3.mem";
            SHIFT1  : integer := 9;
            SHIFT2  : integer := 8
        );
        port (
            clk       : in  std_logic;
            reset     : in  std_logic;
            feat_we   : in  std_logic;
            feat_addr : in  unsigned(10 downto 0);
            feat_data : in  signed(7 downto 0);
            start     : in  std_logic;
            busy      : out std_logic;
            done      : out std_logic;
            class_id  : out unsigned(3 downto 0);
            class_ok  : out std_logic
        );
    end component;

    component class_to_led is
        port (
            class_id : in  unsigned(3 downto 0);
            valid    : in  std_logic;
            ch0      : out std_logic_vector(7 downto 0);
            ch1      : out std_logic_vector(7 downto 0);
            red      : out std_logic_vector(7 downto 0);
            green    : out std_logic_vector(7 downto 0);
            blue     : out std_logic_vector(7 downto 0)
        );
    end component;

    component pwm8 is
        port (
            clk     : in  std_logic;
            duty    : in  std_logic_vector(7 downto 0);
            pwm_out : out std_logic
        );
    end component;

    -- UART receive
    signal rx_data : std_logic_vector(7 downto 0);
    signal rx_dv   : std_logic;

    -- MFCC frame parser -> kws_mlp feature writes + start
    type mstate_t is (M_HUNT, M_INDEX, M_COEFF, M_CKSUM);
    signal mstate      : mstate_t := M_HUNT;
    signal frame_count : unsigned(5 downto 0) := (others => '0');  -- 0..48
    signal coeff_idx   : unsigned(4 downto 0) := (others => '0');  -- 0..19
    signal csum        : unsigned(7 downto 0) := (others => '0');

    signal feat_we_r   : std_logic := '0';
    signal feat_addr_r : unsigned(10 downto 0) := (others => '0');
    signal feat_data_r : signed(7 downto 0) := (others => '0');
    signal mlp_start_r : std_logic := '0';

    -- kws_mlp results
    signal mlp_busy  : std_logic;
    signal mlp_done  : std_logic;
    signal mlp_class : unsigned(3 downto 0);
    signal mlp_ok    : std_logic;

    -- result latch + classification-hold watchdog
    signal class_latch : unsigned(3 downto 0) := (others => '0');
    constant CLS_HOLD  : unsigned(24 downto 0) := to_unsigned(12_000_000, 25); -- ~1 s
    signal   cls_wd    : unsigned(24 downto 0) := (others => '1');
    signal   cls_active : std_logic;

    -- reply transmitter sequencer
    type tstate_t is (T_IDLE, T_S0, T_W0, T_S1, T_W1, T_S2, T_W2);
    signal tstate    : tstate_t := T_IDLE;
    signal reply0    : std_logic_vector(7 downto 0) := (others => '0');
    signal reply1    : std_logic_vector(7 downto 0) := (others => '0');
    signal reply2    : std_logic_vector(7 downto 0) := (others => '0');
    signal tx_start_r : std_logic := '0';
    signal tx_data_r  : std_logic_vector(7 downto 0) := (others => '0');
    signal tx_busy    : std_logic;
    signal tx_line    : std_logic;

    -- class_to_led outputs
    signal cls_ch0, cls_ch1, cls_red, cls_green, cls_blue : std_logic_vector(7 downto 0);

    -- idle rainbow animation
    signal phase       : unsigned(23 downto 0) := (others => '0');
    signal r_idle      : std_logic_vector(7 downto 0) := (others => '0');
    signal g_idle      : std_logic_vector(7 downto 0) := (others => '0');
    signal b_idle      : std_logic_vector(7 downto 0) := (others => '0');
    signal green_idle  : std_logic_vector(7 downto 0) := (others => '0');

    -- button synchroniser
    signal btn_s0, btn_s1 : std_logic_vector(1 downto 0) := (others => '0');

    -- brightness selected into the PWM channels
    signal duty_ld1, duty_ld2          : std_logic_vector(7 downto 0);
    signal duty_r,   duty_g,   duty_b  : std_logic_vector(7 downto 0);
    signal rgb_r_h,  rgb_g_h,  rgb_b_h : std_logic_vector(7 downto 0);

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

    ------------------------------------------------------------------
    -- UART receiver
    ------------------------------------------------------------------
    u_rx : uart_rx
        generic map ( CLKS_PER_BIT => CLKS_PER_BIT )
        port map (
            clk        => sysclk,
            rx         => uart_txd_in,
            data_out   => rx_data,
            data_valid => rx_dv
        );

    ------------------------------------------------------------------
    -- MFCC frame parser: counted framing + checksum.
    -- Writes accepted coefficients straight into the kws_mlp feature memory
    -- and pulses start after a full 49-frame window.
    ------------------------------------------------------------------
    process(sysclk)
    begin
        if rising_edge(sysclk) then
            feat_we_r   <= '0';
            mlp_start_r <= '0';

            if rx_dv = '1' then
                case mstate is
                    when M_HUNT =>
                        if rx_data = x"FE" then
                            mstate <= M_INDEX;
                        end if;

                    when M_INDEX =>
                        csum      <= unsigned(rx_data);      -- seed with frame_index
                        coeff_idx <= (others => '0');
                        mstate    <= M_COEFF;

                    when M_COEFF =>
                        -- write this coefficient into the feature memory
                        if mlp_busy = '0' then
                            feat_addr_r <= to_unsigned(to_integer(frame_count) * 20
                                                       + to_integer(coeff_idx), 11);
                            feat_data_r <= signed(rx_data);
                            feat_we_r   <= '1';
                        end if;
                        csum <= csum + unsigned(rx_data);
                        if coeff_idx = to_unsigned(19, 5) then
                            mstate <= M_CKSUM;
                        else
                            coeff_idx <= coeff_idx + 1;
                        end if;

                    when M_CKSUM =>
                        if unsigned(rx_data) = csum then     -- frame accepted
                            if frame_count = to_unsigned(48, 6) then
                                mlp_start_r <= '1';
                                frame_count <= (others => '0');
                            else
                                frame_count <= frame_count + 1;
                            end if;
                        end if;
                        -- checksum mismatch: silently drop, keep frame_count so a
                        -- single corrupt frame just re-arms on the next window.
                        mstate <= M_HUNT;

                    when others =>
                        mstate <= M_HUNT;
                end case;
            end if;
        end if;
    end process;

    ------------------------------------------------------------------
    -- keyword-spotting MLP (.mem paths come in through the generics above)
    ------------------------------------------------------------------
    u_mlp : kws_mlp
        generic map (
            W1_FILE => W1_FILE,
            B1_FILE => B1_FILE,
            W2_FILE => W2_FILE,
            B2_FILE => B2_FILE,
            W3_FILE => W3_FILE,
            B3_FILE => B3_FILE,
            SHIFT1  => 9,
            SHIFT2  => 8
        )
        port map (
            clk       => sysclk,
            reset     => '0',
            feat_we   => feat_we_r,
            feat_addr => feat_addr_r,
            feat_data => feat_data_r,
            start     => mlp_start_r,
            busy      => mlp_busy,
            done      => mlp_done,
            class_id  => mlp_class,
            class_ok  => mlp_ok
        );

    ------------------------------------------------------------------
    -- Result latch + classification-hold watchdog
    ------------------------------------------------------------------
    process(sysclk)
    begin
        if rising_edge(sysclk) then
            if mlp_done = '1' then
                class_latch <= mlp_class;
                cls_wd      <= (others => '0');
            elsif cls_wd < to_unsigned(2**25 - 1, 25) then
                cls_wd <= cls_wd + 1;
            end if;
        end if;
    end process;
    cls_active <= '1' when cls_wd < CLS_HOLD else '0';

    ------------------------------------------------------------------
    -- Reply transmitter: send 0xCA, class, checksum after each inference
    ------------------------------------------------------------------
    u_tx : uart_tx
        generic map ( CLKS_PER_BIT => CLKS_PER_BIT )
        port map (
            clk      => sysclk,
            tx_start => tx_start_r,
            tx_data  => tx_data_r,
            tx       => tx_line,
            tx_busy  => tx_busy
        );

    process(sysclk)
    begin
        if rising_edge(sysclk) then
            tx_start_r <= '0';
            case tstate is
                when T_IDLE =>
                    if mlp_done = '1' then
                        reply0 <= x"CA";
                        reply1 <= "0000" & std_logic_vector(mlp_class);
                        reply2 <= std_logic_vector(unsigned'(x"CA") + resize(mlp_class, 8));
                        tstate <= T_S0;
                    end if;

                when T_S0 =>
                    if tx_busy = '0' then
                        tx_data_r  <= reply0;
                        tx_start_r <= '1';
                        tstate     <= T_W0;
                    end if;
                when T_W0 =>
                    if tx_busy = '1' then tstate <= T_S1; end if;

                when T_S1 =>
                    if tx_busy = '0' then
                        tx_data_r  <= reply1;
                        tx_start_r <= '1';
                        tstate     <= T_W1;
                    end if;
                when T_W1 =>
                    if tx_busy = '1' then tstate <= T_S2; end if;

                when T_S2 =>
                    if tx_busy = '0' then
                        tx_data_r  <= reply2;
                        tx_start_r <= '1';
                        tstate     <= T_W2;
                    end if;
                when T_W2 =>
                    if tx_busy = '1' then tstate <= T_IDLE; end if;

                when others =>
                    tstate <= T_IDLE;
            end case;
        end if;
    end process;

    uart_rxd_out <= tx_line;

    ------------------------------------------------------------------
    -- Class -> LED color mapping
    ------------------------------------------------------------------
    u_cls_led : class_to_led
        port map (
            class_id => class_latch,
            valid    => cls_active,
            ch0      => cls_ch0,
            ch1      => cls_ch1,
            red      => cls_red,
            green    => cls_green,
            blue     => cls_blue
        );

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
    -- Source select: lamp-test / forced-demo / classification / idle
    ------------------------------------------------------------------
    process(btn_s1, cls_active, cls_ch0, cls_ch1, cls_red, cls_green, cls_blue,
            r_idle, g_idle, b_idle, green_idle)
    begin
        if btn_s1(0) = '1' then                     -- lamp test
            duty_ld1 <= x"FF"; duty_ld2 <= x"FF";
            duty_r   <= x"FF"; duty_g   <= x"FF"; duty_b <= x"FF";
        elsif btn_s1(1) = '1' then                  -- forced idle rainbow
            duty_ld1 <= green_idle; duty_ld2 <= green_idle;
            duty_r   <= r_idle; duty_g <= g_idle; duty_b <= b_idle;
        elsif cls_active = '1' then                 -- on-FPGA classification
            duty_ld1 <= cls_ch0; duty_ld2 <= cls_ch1;
            duty_r   <= cls_red; duty_g <= cls_green; duty_b <= cls_blue;
        else                                        -- no recent class: idle rainbow
            duty_ld1 <= green_idle; duty_ld2 <= green_idle;
            duty_r   <= r_idle; duty_g <= g_idle; duty_b <= b_idle;
        end if;
    end process;

    -- Cap RGB brightness at ~75% (duty*3/4 = duty/2 + duty/4).
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
