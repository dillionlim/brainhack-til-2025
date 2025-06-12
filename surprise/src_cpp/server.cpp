#include "SurpriseManager.h"
#include <crow.h>
#include "base64.h"
#include <vector>
#include <utility>
#include <iostream>

int main() {
    crow::App<> app;
    surprise::SurpriseManager manager;

    // app.loglevel(crow::LogLevel::Warning);

    CROW_ROUTE(app, "/health")
    .methods(crow::HTTPMethod::GET)
    ([]{
        crow::response res;
        res.code = 200;
        res.set_header("Content-Type", "application/json");
        res.body = R"({"message":"health ok"})";
        return res;
    });

    CROW_ROUTE(app, "/surprise")
    .methods(crow::HTTPMethod::POST)
    ([&](const crow::request& req){
        auto x = crow::json::load(req.body);
        if (!x || !x.has("instances")) {
            crow::response err{400, R"({"error":"invalid JSON"})"};
            err.set_header("Content-Type", "application/json");
            return err;
        }

        crow::json::wvalue out;
        out["predictions"] = crow::json::wvalue::list();
        size_t outer_idx = 0;

        for (auto& inst : x["instances"]) {
            std::vector<std::vector<uchar>> slices;
            for (auto& sl : inst["slices"]) {
                auto decoded = base64_decode(sl.s());
                slices.emplace_back(decoded.begin(), decoded.end());
            }

            auto perm = manager.surprise(slices);
            std::cout << "[";
            for (size_t i = 0; i < perm.size(); ++i) {
                std::cout << perm[i];
                if (i + 1 < perm.size()) 
                    std::cout << ",";
            }
            std::cout << "]\n";

            crow::json::wvalue p;
            for (size_t i = 0; i < perm.size(); ++i) {
                p[i] = static_cast<int>(perm[i]);
            }

            out["predictions"][outer_idx++] = std::move(p);
        }

        crow::response res;
        res.code = 200;
        res.set_header("Content-Type", "application/json");
        res.body = out.dump();
        return res;
    });

    app.port(5005).multithreaded().run();
    return 0;
}
