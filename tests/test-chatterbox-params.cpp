// test-chatterbox-params.cpp — unit tests for chatterbox_context_params defaults.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "chatterbox.h"

TEST_CASE("chatterbox_params: default values are sensible", "[unit][chatterbox]") {
    struct chatterbox_context_params p = chatterbox_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
    REQUIRE(p.temperature == Catch::Approx(0.8f));
    REQUIRE(p.cfg_weight == Catch::Approx(0.5f));
    REQUIRE(p.exaggeration == Catch::Approx(0.5f));
    REQUIRE(p.repetition_penalty == Catch::Approx(1.2f));
    REQUIRE(p.min_p == Catch::Approx(0.05f));
    REQUIRE(p.top_p == Catch::Approx(1.0f));
    REQUIRE(p.max_speech_tokens == 1000);
    REQUIRE(p.cfm_steps == 10);
}
