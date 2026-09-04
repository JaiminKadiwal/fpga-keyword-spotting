library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use STD.TEXTIO.ALL;
use IEEE.STD_LOGIC_TEXTIO.ALL;

-- Fixed-point keyword-spotting MLP: 980 -> 128 -> 64 -> 12.
--
-- int8 activations, int8 weights, int32 accumulators. Hidden layers apply
-- ReLU then a round-half-up arithmetic right shift (per-layer, from
-- quant_params.json) and clamp to int8 [0, 127]. The output layer keeps int32
-- logits and reports argmax. This datapath must match the Python integer
-- reference in pc/export_weights.py bit-for-bit (see docs/fixed_point_contract.md).
--
-- One multiply-accumulate per clock. Weights/biases are loaded from the .mem
-- files at elaboration; the feature vector is written in through feat_*.
entity kws_mlp is
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
        feat_we   : in  std_logic;                     -- write one feature
        feat_addr : in  unsigned(10 downto 0);         -- 0 .. 979
        feat_data : in  signed(7 downto 0);
        start     : in  std_logic;                     -- begin inference
        busy      : out std_logic;
        done      : out std_logic;                     -- one-clk strobe when class_id valid
        class_id  : out unsigned(3 downto 0);
        class_ok  : out std_logic
    );
end kws_mlp;

architecture rtl of kws_mlp is

    constant N1_IN  : integer := 980;
    constant N1_OUT : integer := 128;
    constant N2_IN  : integer := 128;
    constant N2_OUT : integer := 64;
    constant N3_IN  : integer := 64;
    constant N3_OUT : integer := 12;

    type i8_array  is array (natural range <>) of signed(7 downto 0);
    type i32_array is array (natural range <>) of signed(31 downto 0);

    impure function load_i32(fname : string; n : natural) return i32_array is
        file     ff : text open read_mode is fname;
        variable l  : line;
        variable v  : std_logic_vector(31 downto 0);
        variable r  : i32_array(0 to n - 1);
    begin
        for k in 0 to n - 1 loop
            readline(ff, l);
            hread(l, v);
            r(k) := signed(v);
        end loop;
        return r;
    end function;

    -- Weight ROMs are Verilog $readmemh block RAMs (see weight_rom.v): the VHDL
    -- textio init loop would trip the synthesis loop-iteration limit at L1's
    -- 128*980 = 125,440 entries. Row-major [out][in]: index = j*N_in + i.
    constant W1_DEPTH : integer := N1_OUT * N1_IN;   -- 125440, addr 17 bits
    constant W2_DEPTH : integer := N2_OUT * N2_IN;   -- 8192,   addr 13 bits
    constant W3_DEPTH : integer := N3_OUT * N3_IN;   -- 768,    addr 10 bits

    component weight_rom is
        generic (
            DEPTH     : integer;
            AWIDTH    : integer;
            INIT_FILE : string
        );
        port (
            clk  : in  std_logic;
            addr : in  std_logic_vector(AWIDTH - 1 downto 0);
            dout : out signed(7 downto 0)
        );
    end component;

    -- Shared read index (active layer) and per-ROM addresses / registered reads.
    signal rom_idx : unsigned(16 downto 0) := (others => '0');
    signal addr1   : std_logic_vector(16 downto 0);
    signal addr2   : std_logic_vector(12 downto 0);
    signal addr3   : std_logic_vector(9 downto 0);
    signal dout1, dout2, dout3 : signed(7 downto 0);
    signal w_sel   : signed(7 downto 0);

    -- Bias ROMs are small (128/64/12 entries); constants are fine here.
    constant B1 : i32_array(0 to N1_OUT - 1) := load_i32(B1_FILE, N1_OUT);
    constant B2 : i32_array(0 to N2_OUT - 1) := load_i32(B2_FILE, N2_OUT);
    constant B3 : i32_array(0 to N3_OUT - 1) := load_i32(B3_FILE, N3_OUT);

    -- Activation storage.
    signal feat : i8_array(0 to N1_IN - 1)  := (others => (others => '0'));
    signal h1   : i8_array(0 to N1_OUT - 1) := (others => (others => '0'));
    signal h2   : i8_array(0 to N2_OUT - 1) := (others => (others => '0'));

    -- Two-cycle MAC: S_MAC_SET registers the weight/activation for (layer,j,i)
    -- (synchronous BRAM read), S_MAC_ACC consumes them one cycle later.
    type state_t is (S_IDLE, S_INIT_NEURON, S_MAC_SET, S_MAC_ACC, S_STORE, S_FINISH);
    signal state : state_t := S_IDLE;

    signal layer : integer range 1 to 3 := 1;
    signal j     : integer range 0 to 127 := 0;   -- output neuron index
    signal i     : integer range 0 to 1023 := 0;  -- input index
    signal acc   : signed(31 downto 0) := (others => '0');

    -- Registered activation aligned to the ROM's one-cycle read latency.
    signal x_q : signed(7 downto 0) := (others => '0');

    signal best_val : signed(31 downto 0) := (others => '0');
    signal best_idx : integer range 0 to 15 := 0;

    signal busy_r     : std_logic := '0';
    signal done_r     : std_logic := '0';
    signal class_id_r : unsigned(3 downto 0) := (others => '0');
    signal class_ok_r : std_logic := '0';

    -- Layer geometry helpers.
    function n_in_of(l : integer) return integer is
    begin
        if l = 1 then return N1_IN; elsif l = 2 then return N2_IN; else return N3_IN; end if;
    end function;

    function n_out_of(l : integer) return integer is
    begin
        if l = 1 then return N1_OUT; elsif l = 2 then return N2_OUT; else return N3_OUT; end if;
    end function;

    function bias_of(l, jj : integer) return signed is
    begin
        if l = 1 then return B1(jj); elsif l = 2 then return B2(jj); else return B3(jj); end if;
    end function;

    -- ReLU + round-half-up right shift + clamp to int8 [0, 127].
    function requant(a : signed(31 downto 0); shift : integer) return signed is
        variable v : signed(31 downto 0);
    begin
        v := a;
        if v < 0 then
            v := (others => '0');
        end if;
        if shift > 0 then
            v := shift_right(v + to_signed(2 ** (shift - 1), 32), shift);
        end if;
        if v > 127 then
            v := to_signed(127, 32);
        end if;
        return resize(v, 8);
    end function;

    function shift_of(l : integer) return integer is
    begin
        if l = 1 then return SHIFT1; else return SHIFT2; end if;
    end function;

begin

    busy     <= busy_r;
    done     <= done_r;
    class_id <= class_id_r;
    class_ok <= class_ok_r;

    ------------------------------------------------------------------
    -- Weight ROMs (Verilog $readmemh block RAMs). Addressed by the active
    -- layer's flat index; only the selected layer's index is in range, so the
    -- lower bits feed the smaller ROMs (their outputs are ignored otherwise).
    ------------------------------------------------------------------
    rom_idx <= to_unsigned(j * n_in_of(layer) + i, 17);
    addr1   <= std_logic_vector(rom_idx);
    addr2   <= std_logic_vector(rom_idx(12 downto 0));
    addr3   <= std_logic_vector(rom_idx(9 downto 0));

    rom1 : weight_rom
        generic map ( DEPTH => W1_DEPTH, AWIDTH => 17, INIT_FILE => W1_FILE )
        port map ( clk => clk, addr => addr1, dout => dout1 );
    rom2 : weight_rom
        generic map ( DEPTH => W2_DEPTH, AWIDTH => 13, INIT_FILE => W2_FILE )
        port map ( clk => clk, addr => addr2, dout => dout2 );
    rom3 : weight_rom
        generic map ( DEPTH => W3_DEPTH, AWIDTH => 10, INIT_FILE => W3_FILE )
        port map ( clk => clk, addr => addr3, dout => dout3 );

    -- Select the active layer's ROM output (valid in S_MAC_ACC).
    w_sel <= dout1 when layer = 1 else
             dout2 when layer = 2 else
             dout3;

    process(clk)
    begin
        if rising_edge(clk) then
            done_r <= '0';

            -- Register the activation for (layer, j, i) so it lines up with the
            -- weight ROM's one-cycle read latency. Gated to S_MAC_SET, where i is
            -- guaranteed in range for the active layer; x_q and the ROM output
            -- w_sel are then both valid in the following S_MAC_ACC cycle.
            if state = S_MAC_SET then
                case layer is
                    when 1      => x_q <= feat(i);
                    when 2      => x_q <= h1(i);
                    when others => x_q <= h2(i);
                end case;
            end if;

            if reset = '1' then
                state      <= S_IDLE;
                busy_r     <= '0';
                class_ok_r <= '0';
            else
                case state is
                    when S_IDLE =>
                        busy_r <= '0';
                        if feat_we = '1' then
                            feat(to_integer(feat_addr)) <= feat_data;
                        end if;
                        if start = '1' then
                            busy_r   <= '1';
                            layer    <= 1;
                            j        <= 0;
                            state    <= S_INIT_NEURON;
                        end if;

                    when S_INIT_NEURON =>
                        acc <= resize(bias_of(layer, j), 32);
                        i   <= 0;
                        state <= S_MAC_SET;

                    when S_MAC_SET =>
                        -- w_q/x_q for (layer, j, i) latch this cycle.
                        state <= S_MAC_ACC;

                    when S_MAC_ACC =>
                        acc <= acc + resize(x_q * w_sel, 32);
                        if i = n_in_of(layer) - 1 then
                            state <= S_STORE;
                        else
                            i     <= i + 1;
                            state <= S_MAC_SET;
                        end if;

                    when S_STORE =>
                        if layer = 1 then
                            h1(j) <= requant(acc, shift_of(1));
                        elsif layer = 2 then
                            h2(j) <= requant(acc, shift_of(2));
                        else
                            -- Output layer: track argmax of int32 logits.
                            if j = 0 or acc > best_val then
                                best_val <= acc;
                                best_idx <= j;
                            end if;
                        end if;

                        if j = n_out_of(layer) - 1 then
                            if layer = 3 then
                                state <= S_FINISH;
                            else
                                layer <= layer + 1;
                                j     <= 0;
                                state <= S_INIT_NEURON;
                            end if;
                        else
                            j     <= j + 1;
                            state <= S_INIT_NEURON;
                        end if;

                    when S_FINISH =>
                        class_id_r <= to_unsigned(best_idx, 4);
                        class_ok_r <= '1';
                        done_r     <= '1';
                        busy_r     <= '0';
                        state      <= S_IDLE;

                    when others =>
                        state <= S_IDLE;
                end case;
            end if;
        end if;
    end process;

end rtl;
