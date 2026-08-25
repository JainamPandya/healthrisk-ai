const form = document.getElementById("riskForm");

const resultSection = document.getElementById("result");
const riskScore = document.getElementById("riskScore");
const riskCategory = document.getElementById("riskCategory");

const increasingFactors =
    document.getElementById("increasingFactors");

const decreasingFactors =
    document.getElementById("decreasingFactors");


form.addEventListener("submit", async function (event) {

    event.preventDefault();

    const patientData = {

        // Patient information
        race: document.getElementById("race").value,
        gender: document.getElementById("gender").value,
        age: document.getElementById("age").value,
        weight: document.getElementById("weight").value,


        // Hospitalization
        admission_type_id:
            Number(document.getElementById("admission_type_id").value),

        discharge_disposition_id:
            Number(document.getElementById("discharge_disposition_id").value),

        admission_source_id:
            Number(document.getElementById("admission_source_id").value),

        time_in_hospital:
            Number(document.getElementById("time_in_hospital").value),

        payer_code:
            document.getElementById("payer_code").value,

        medical_specialty:
            document.getElementById("medical_specialty").value,


        // Clinical information
        num_lab_procedures:
            Number(document.getElementById("num_lab_procedures").value),

        num_procedures:
            Number(document.getElementById("num_procedures").value),

        num_medications:
            Number(document.getElementById("num_medications").value),

        number_outpatient:
            Number(document.getElementById("number_outpatient").value),

        number_emergency:
            Number(document.getElementById("number_emergency").value),

        number_inpatient:
            Number(document.getElementById("number_inpatient").value),

        diag_1:
            document.getElementById("diag_1").value,

        diag_2:
            document.getElementById("diag_2").value,

        diag_3:
            document.getElementById("diag_3").value,

        number_diagnoses:
            Number(document.getElementById("number_diagnoses").value),

        max_glu_serum:
            document.getElementById("max_glu_serum").value,

        A1Cresult:
            document.getElementById("A1Cresult").value,


        // Diabetes medications

        metformin:
            document.getElementById("metformin").value,

        repaglinide: "No",

        nateglinide: "No",

        chlorpropamide: "No",

        glimepiride: "No",

        acetohexamide: "No",

        glipizide:
            document.getElementById("glipizide").value,

        glyburide:
            document.getElementById("glyburide").value,

        tolbutamide: "No",

        pioglitazone: "No",

        rosiglitazone: "No",

        acarbose: "No",

        miglitol: "No",

        troglitazone: "No",

        tolazamide: "No",

        examide: "No",

        citoglipton: "No",

        insulin:
            document.getElementById("insulin").value,


        // Combination medications
        // IMPORTANT:
        // These names must match PatientData.

        glyburide_metformin: "No",

        glipizide_metformin: "No",

        glimepiride_pioglitazone: "No",

        metformin_rosiglitazone: "No",

        metformin_pioglitazone: "No",


        // Other treatment information
        change:
            document.getElementById("change").value,

        diabetesMed:
            document.getElementById("diabetesMed").value
    };


    console.log("Sending patient data:", patientData);


    try {

        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(patientData)

        });


        if (!response.ok) {

            const errorText = await response.text();

            console.error(
                "API error:",
                response.status,
                errorText
            );

            throw new Error(
                `Prediction failed (${response.status})`
            );
        }


        const data = await response.json();


        console.log("Prediction result:", data);


        // Show result section

        resultSection.classList.remove("hidden");


        // Risk score

        riskScore.textContent =
            (data.risk_score * 100).toFixed(1) + "%";


        // Risk category

        riskCategory.textContent =
            data.risk_category;


        // Clear previous explanations

        increasingFactors.innerHTML = "";

        decreasingFactors.innerHTML = "";


        // Increasing factors

        data.factors_increasing.forEach(function (item) {

            const li =
                document.createElement("li");

            li.textContent =
                `↑ ${item.feature} (+${item.shap_value.toFixed(4)})`;

            increasingFactors.appendChild(li);

        });


        // Decreasing factors

        data.factors_decreasing.forEach(function (item) {

            const li =
                document.createElement("li");

            li.textContent =
                `↓ ${item.feature} (${item.shap_value.toFixed(4)})`;

            decreasingFactors.appendChild(li);

        });


        // Scroll to result

        resultSection.scrollIntoView({
            behavior: "smooth"
        });

    }

    catch (error) {

        console.error(error);

        alert(
            "Prediction failed. Check the FastAPI terminal for details."
        );

    }

});