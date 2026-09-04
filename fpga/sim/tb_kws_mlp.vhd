library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use STD.TEXTIO.ALL;

-- Phase 2B: replay the Python fixed-point golden vectors through kws_mlp and
-- check that class_id matches fp_predicted_class for every vector.
--   golden_features.txt : decimal int8, one per line, 980 per vector
--   golden_labels.txt   : first line = vector count, then one class per line
entity tb_kws_mlp is
    generic (
        FEAT_FILE  : string := "fpga/sim/vectors/golden_features.txt";
        LABEL_FILE : string := "fpga/sim/vectors/golden_labels.txt";
        W1_FILE    : string := "models/exported/weights_l1.mem";
        B1_FILE    : string := "models/exported/bias_l1.mem";
        W2_FILE    : string := "models/exported/weights_l2.mem";
        B2_FILE    : string := "models/exported/bias_l2.mem";
        W3_FILE    : string := "models/exported/weights_l3.mem";
        B3_FILE    : string := "models/exported/bias_l3.mem";
        SHIFT1     : integer := 9;
        SHIFT2     : integer := 8
    );
end tb_kws_mlp;

architecture sim of tb_kws_mlp is
    constant N_IN : integer := 980;

    signal clk       : std_logic := '0';
    signal reset     : std_logic := '1';
    signal feat_we   : std_logic := '0';
    signal feat_addr : unsigned(10 downto 0) := (others => '0');
    signal feat_data : signed(7 downto 0) := (others => '0');
    signal start     : std_logic := '0';
    signal busy      : std_logic;
    signal done      : std_logic;
    signal class_id  : unsigned(3 downto 0);
    signal class_ok  : std_logic;

    signal sim_done : boolean := false;
begin

    dut : entity work.kws_mlp
        generic map (
            W1_FILE => W1_FILE, B1_FILE => B1_FILE,
            W2_FILE => W2_FILE, B2_FILE => B2_FILE,
            W3_FILE => W3_FILE, B3_FILE => B3_FILE,
            SHIFT1 => SHIFT1, SHIFT2 => SHIFT2
        )
        port map (
            clk => clk, reset => reset,
            feat_we => feat_we, feat_addr => feat_addr, feat_data => feat_data,
            start => start, busy => busy, done => done,
            class_id => class_id, class_ok => class_ok
        );

    clk <= not clk after 5 ns when not sim_done else '0';

    stimulus : process
        file feats  : text open read_mode is FEAT_FILE;
        file labels : text open read_mode is LABEL_FILE;
        variable l        : line;
        variable n_vec    : integer;
        variable fval     : integer;
        variable expected : integer;
        variable got      : integer;
        variable passes   : integer := 0;
        variable fails    : integer := 0;
        variable msg      : line;
    begin
        -- Reset for a few cycles.
        reset <= '1';
        wait until rising_edge(clk);
        wait until rising_edge(clk);
        reset <= '0';
        wait until rising_edge(clk);

        readline(labels, l);
        read(l, n_vec);
        report "tb_kws_mlp: replaying " & integer'image(n_vec) & " golden vectors";

        for v in 0 to n_vec - 1 loop
            -- Load one 980-element feature vector, one value per clock.
            for k in 0 to N_IN - 1 loop
                readline(feats, l);
                read(l, fval);
                feat_addr <= to_unsigned(k, 11);
                feat_data <= to_signed(fval, 8);
                feat_we   <= '1';
                wait until rising_edge(clk);
            end loop;
            feat_we <= '0';

            readline(labels, l);
            read(l, expected);

            -- Trigger inference and wait for completion.
            start <= '1';
            wait until rising_edge(clk);
            start <= '0';
            wait until done = '1';
            got := to_integer(class_id);

            if got = expected then
                passes := passes + 1;
            else
                fails := fails + 1;
                write(msg, string'("MISMATCH vec="));
                write(msg, v);
                write(msg, string'(" expected="));
                write(msg, expected);
                write(msg, string'(" got="));
                write(msg, got);
                report msg.all severity warning;
                deallocate(msg);
            end if;

            wait until rising_edge(clk);
        end loop;

        report "tb_kws_mlp RESULT: " & integer'image(passes) & "/"
            & integer'image(passes + fails) & " vectors match the Python golden class";
        if fails = 0 then
            report "tb_kws_mlp: PASS - VHDL matches Python fixed-point bit-for-bit" severity note;
        else
            report "tb_kws_mlp: FAIL - " & integer'image(fails) & " mismatches" severity error;
        end if;

        sim_done <= true;
        wait;
    end process;

end sim;
